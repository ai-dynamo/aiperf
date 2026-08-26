// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Resolve normalized profile [`Inputs`] into a [`BenchmarkRun`].
//!
//! The CLI authoring layer (flag and YAML constructors) normalizes to [`Inputs`]
//! and calls [`resolve`], keeping wire defaults in one place. This resolution was
//! moved verbatim out of `aiperf-cli`'s `load.rs` so the runtime owns the
//! `Inputs` -> `BenchmarkRun` projection; the CLI re-exports the moved types so
//! its existing call sites are unchanged.

use std::path::PathBuf;

use crate::config::model::artifacts::Artifacts;
use crate::config::model::config::Metadata;
use crate::config::model::dataset::{
    AudioSpec, Dataset, Distribution, ImageSpec, PrefixPrompts, PromptSelection, Prompts,
    RecordedAgentGraphConfig, Sampling, Synthetic, VideoSpec,
};
use crate::config::model::endpoint::{
    ConnectionReuse, Endpoint, EndpointType, RequestContentType, ResetKvCacheConfig,
    ServerProfilerConfig, WaitForModelMode,
};
use crate::config::model::metrics::Metrics;
use crate::config::model::models::{ModelItem, ModelStrategy, Models};
use crate::config::model::phase::{AdaptiveScale, Phase, PhaseCommon, PhaseKind, PhaseRole};
use crate::config::model::rate_series::RateSeries;
use crate::config::model::runtime::Runtime;
use crate::config::model::tokenizer::Tokenizer;
use crate::config::model::{BenchmarkConfig, BenchmarkRun, Resolved};
use crate::config::model::{DispatchMode, HopRouting};
use crate::config::phase_validate::{
    DEFAULT_BENCHMARK_GRACE_PERIOD_SECONDS, apply_cli_loadgen_overlays,
    normalize_and_validate_phases,
};

/// Exact-match placeholder words used as an entire model name.
const FAKE_MODEL_EXACT: &[&str] = &[
    "test",
    "mock",
    "fake",
    "dummy",
    "example",
    "sample",
    "placeholder",
];

/// Placeholder markers requiring separators to avoid matching real names.
const FAKE_MODEL_SUBSTRINGS: &[&str] = &[
    "mock-",
    "-mock",
    "fake-",
    "-fake",
    "test-model",
    "-test-model",
    "your-model",
    "my-model",
    "model-name",
    "model-id",
];
/// Return whether `name` is a placeholder rather than a model identifier.
fn is_fake_model_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    if name.contains('/') || name.contains('\\') || name.starts_with('.') || name.starts_with('~') {
        return false;
    }
    let normalized = name.to_lowercase().replace('_', "-");
    if FAKE_MODEL_EXACT.contains(&normalized.as_str()) {
        return true;
    }
    FAKE_MODEL_SUBSTRINGS.iter().any(|s| normalized.contains(s))
}

const DEFAULT_TIMEOUT_SECONDS: f64 = 21600.0;
const DEFAULT_CONNECTION_LIMIT: u32 = 2500;
const DEFAULT_KEEPALIVE_TIMEOUT: f64 = 300.0;
const DEFAULT_WAIT_FOR_MODEL_INTERVAL: f64 = 5.0;
/// Default request bound when no count/duration/schedule bounds the run.
const DEFAULT_REQUEST_COUNT: u64 = 10;

fn authored_prompt_selection(corpus: Option<&str>) -> Option<PromptSelection> {
    corpus.map(|corpus| PromptSelection {
        corpus: Some(corpus.to_string()),
    })
}

/// Parse `--cache-bust` / `Inputs.cache_bust` into the typed wire policy.
fn cache_bust_from_inputs(inputs: &Inputs) -> Option<crate::config::model::dataset::CacheBust> {
    use crate::config::model::dataset::{CacheBust, CacheBustTarget};
    let raw = inputs.cache_bust.as_deref()?;
    let target = match raw {
        "none" => return None,
        "system_prefix" => CacheBustTarget::SystemPrefix,
        "system_suffix" => CacheBustTarget::SystemSuffix,
        "first_turn_prefix" => CacheBustTarget::FirstTurnPrefix,
        "first_turn_suffix" => CacheBustTarget::FirstTurnSuffix,
        "warmup_isolation_system" => CacheBustTarget::WarmupIsolationSystem,
        "warmup_isolation_first_turn" => CacheBustTarget::WarmupIsolationFirstTurn,
        other => {
            tracing::warn!(target = %other, "unknown cache_bust target; ignoring");
            return None;
        }
    };
    Some(CacheBust { target })
}

/// Merge CLI/YAML synthesis knobs onto `Inputs.synthesis` for the dataset wire.
fn resolve_synthesis(inputs: &Inputs) -> Option<serde_json::Value> {
    let mut map = match &inputs.synthesis {
        Some(serde_json::Value::Object(m)) => m.clone(),
        Some(other) => {
            let mut m = serde_json::Map::new();
            m.insert("_base".into(), other.clone());
            m
        }
        None => serde_json::Map::new(),
    };
    let mut touched = inputs.synthesis.is_some();
    if let Some(wrap) = inputs.allow_dataset_wrap {
        map.insert("allow_dataset_wrap".into(), serde_json::Value::Bool(wrap));
        touched = true;
    }
    if let Some(cap) = inputs.trace_idle_gap_cap_seconds
        && let Some(n) = serde_json::Number::from_f64(cap)
    {
        map.insert("idle_gap_cap_seconds".into(), serde_json::Value::Number(n));
        touched = true;
    }
    if let Some(v) = inputs.max_context_length {
        map.insert("max_context_length".into(), serde_json::Value::from(v));
        touched = true;
    }
    if let Some(target) = &inputs.cache_bust {
        map.insert(
            "cache_bust_target".into(),
            serde_json::Value::String(target.clone()),
        );
        touched = true;
    }
    if inputs.use_think_time_only {
        map.insert("use_think_time_only".into(), serde_json::Value::Bool(true));
        touched = true;
    }
    if inputs.burst_phase_starts {
        map.insert("burst_phase_starts".into(), serde_json::Value::Bool(true));
        touched = true;
    }
    touched.then(|| serde_json::Value::Object(map))
}

/// Normalize `--profile-export-prefix` like Python `ArtifactsConfig`: strip a
/// directory component and known export suffixes, defaulting to `profile_export`.
fn artifact_export_stem(prefix: Option<&str>) -> String {
    const SUFFIXES: &[&str] = &[
        "_aiperf_timeslices.csv",
        "_aiperf_timeslices.json",
        "_aiperf.csv",
        "_aiperf.json",
        "_records.csv",
        "_raw.jsonl",
        "_server_metrics.parquet",
        "_server_metrics.jsonl",
        "_server_metrics.json",
        "_server_metrics.csv",
        "_gpu_telemetry.jsonl",
        ".jsonl",
        ".parquet",
        ".csv",
        ".json",
    ];
    let Some(raw) = prefix.map(str::trim).filter(|s| !s.is_empty()) else {
        return "profile_export".to_string();
    };
    let name = std::path::Path::new(raw)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(raw);
    let mut stem = name.to_string();
    for suffix in SUFFIXES {
        if let Some(stripped) = stem.strip_suffix(suffix) {
            stem = stripped.to_string();
            break;
        }
    }
    if stem.is_empty() {
        "profile_export".to_string()
    } else {
        stem
    }
}

/// A leading warmup phase's axes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Warmup {
    /// Warmup concurrency (inherits profiling concurrency when `None`).
    pub concurrency: Option<u32>,
    /// Warmup request rate (Poisson when set).
    pub rate: Option<f64>,
    /// Warmup request bound.
    pub requests: Option<u64>,
    /// Warmup session bound.
    pub sessions: Option<u64>,
    /// Warmup prefill concurrency.
    pub prefill_concurrency: Option<u32>,
    /// Warmup arrival distribution for `rate` (`poisson`/`gamma`/`constant`).
    pub rate_mode: Option<String>,
    /// Warmup concurrency-ramp duration.
    pub concurrency_ramp: Option<f64>,
    /// Warmup rate-ramp duration.
    pub rate_ramp: Option<f64>,
    /// Warmup prefill-concurrency-ramp duration.
    pub prefill_ramp: Option<f64>,
    /// Warmup duration bound.
    pub duration: Option<f64>,
    /// Warmup grace period.
    pub grace_period: Option<f64>,
}

/// Normalized profile inputs.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Inputs {
    pub model_names: Vec<String>,
    pub urls: Vec<String>,
    pub endpoint_type: String,
    pub transport: crate::config::model::transport::Transport,
    pub streaming: bool,
    pub timeout_seconds: Option<f64>,
    pub use_legacy_max_tokens: bool,
    pub use_server_token_count: bool,
    pub download_video_content: bool,
    /// Extra request-body inputs (endpoint.extra).
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// Custom server-metrics scrape URLs.
    pub server_metrics_urls: Vec<String>,
    pub connection_reuse: Option<ConnectionReuse>,
    /// Verify TLS peer certificates. Absent leaves the `Endpoint` default (on).
    pub ssl_verify: Option<bool>,
    /// Unix-domain socket to dial instead of the URL host.
    pub uds_path: Option<String>,
    pub request_content_type: Option<RequestContentType>,
    pub wait_for_model_timeout: Option<f64>,
    pub wait_for_model_mode: Option<WaitForModelMode>,
    pub wait_for_model_interval: Option<f64>,
    pub apply_chat_template: bool,
    pub prefill_concurrency: Option<u32>,
    pub prefill_ramp: Option<f64>,
    pub gpu_telemetry_enabled: bool,
    /// Collector backend id (`dcgm`, `pynvml`, or `amdsmi`). Absent selects the
    /// `dcgm` default.
    #[serde(default)]
    pub gpu_telemetry_collector: Option<String>,
    /// Custom DCGM URLs.
    pub gpu_telemetry_urls: Vec<String>,
    /// Custom DCGM metrics CSV path (`--gpu-telemetry <file>.csv`).
    pub gpu_telemetry_metrics_file: Option<String>,
    pub server_metrics_enabled: bool,
    pub server_metrics_formats: Option<Vec<String>>,
    /// Goodput SLO thresholds (metric -> threshold ms).
    pub slos: serde_json::Map<String, serde_json::Value>,
    /// Fixed mean network RTT, milliseconds.
    pub network_latency_mean: Option<f64>,
    /// Automatic RTT probe with an optional ping interval (seconds).
    pub network_latency_probe: Option<f64>,
    /// OTLP collector URL (`--otel-url`).
    pub otel_url: Option<String>,
    /// GenAI provider label (`--gen-ai-provider`).
    pub otel_provider: Option<String>,
    /// Extra OTLP resource attributes (`--otel-resource-attributes`).
    pub otel_resource_attributes: Vec<(String, String)>,
    /// MLflow sink params.
    pub mlflow: crate::config::model::export::MlflowParams,
    /// W&B sink params.
    pub wandb: crate::config::model::export::WandbParams,
    pub api_key: Option<String>,
    pub headers: std::collections::BTreeMap<String, String>,
    pub tokenizer_name: Option<String>,
    pub tokenizer_revision: Option<String>,
    pub tokenizer_trust: bool,
    pub server_tokenizer_url: Option<String>,
    pub isl: Distribution,
    pub osl: Option<Distribution>,
    /// Turns-per-session distribution (multi-turn).
    pub turns: Option<Distribution>,
    /// Per-session think-time delay ratio.
    pub turn_delay_ratio: f64,
    /// Inter-turn fixed delay distribution, milliseconds.
    pub turn_delay_ms: Option<Distribution>,
    /// Per-session affinity header name (`endpoint.session_header`).
    pub session_header: Option<String>,
    /// Explicit forward-proxy URL for benchmark traffic (`--proxy`).
    pub proxy: Option<String>,
    /// Honor the ambient proxy environment for benchmark traffic
    /// (`--proxy-from-env`).
    pub proxy_from_env: bool,
    /// Custom request path appended to the endpoint URL (`endpoint.path`).
    pub endpoint_path: Option<String>,
    /// Optional reset-KV-cache hook policy (`endpoint.reset_kv_cache`).
    pub reset_kv_cache: Option<ResetKvCacheConfig>,
    /// Optional server-profiler hook policy (`endpoint.server_profiler`).
    pub server_profiler: Option<ServerProfilerConfig>,
    /// Per-record export formats (`artifacts.records`; default `["jsonl"]`,
    /// empty = summary-only).
    pub records_formats: Vec<String>,
    /// Summary-export formats (`artifacts.summary`); empty = unauthored, both
    /// `json` and `csv` ship.
    pub summary_formats: Vec<String>,
    /// Files to materialize into the run directory (`artifacts.user_files`).
    pub user_files: Vec<crate::config::model::artifacts::UserFile>,
    /// Emit the raw request/response JSONL (`artifacts.raw`).
    pub export_raw: bool,
    /// Emit per-request HTTP trace columns (`artifacts.trace`).
    pub export_trace: bool,
    /// Emit the per-request outputs JSON (`artifacts.export_outputs_json`).
    pub export_outputs_json: bool,
    /// Show per-request HTTP trace timing in the console (`--show-trace-timing`).
    pub show_trace_timing: bool,
    /// Base filename stem for exported artifacts (`--profile-export-prefix`).
    pub profile_export_prefix: Option<String>,
    /// Weka: emit turn delays from recorded think_time only (`--use-think-time-only`).
    pub use_think_time_only: bool,
    /// Maximum peak prompt+output context length for Weka traces.
    pub max_context_length: Option<u32>,
    /// Allow dataset wrap when concurrency exceeds the loaded pool.
    /// `None` leaves the engine default; `Some(false)` matches Python default.
    pub allow_dataset_wrap: Option<bool>,
    /// Cache-bust target snake_case name (`none` / `first_turn_prefix` / …).
    pub cache_bust: Option<String>,
    /// AGENTIC_REPLAY synchronized phase-start bursts.
    pub burst_phase_starts: bool,
    /// Per-trace idle-gap cap, seconds (`--trace-idle-gap-cap-seconds`).
    pub trace_idle_gap_cap_seconds: Option<f64>,
    /// Global system-idle cap, seconds (`--system-idle-gap-cap-seconds`). Legacy Weka only.
    pub system_idle_gap_cap_seconds: Option<f64>,
    /// HuggingFace repo for generic Weka loader (`--hf-weka-dataset`).
    pub hf_weka_dataset: Option<String>,
    /// Baseten whole-session sample ratio (`--trace-session-sample-ratio`).
    pub trace_session_sample_ratio: Option<f64>,
    /// Agentic warmup barrier grace (`--agentic-warmup-grace-period`).
    pub agentic_warmup_grace_period: Option<f64>,
    /// Abort-on-failure-ratio threshold (`--failed-request-threshold`).
    pub failed_request_threshold: Option<f64>,
    /// Mixed ISL/OSL sequence distribution (`--seq-dist`).
    pub sequence_distribution: Option<Vec<crate::config::model::dataset::SeqDistEntry>>,
    pub batch_size: u32,
    pub sampling: String,
    pub entries: u32,
    /// Explicit entry count for file/public datasets (None when defaulted).
    pub dataset_entries: Option<u32>,
    /// Profiling-phase session bound (from `num_conversations`).
    pub sessions: Option<u64>,
    pub concurrency: Option<u32>,
    pub request_rate: Option<f64>,
    /// Arrival distribution for `request_rate` (`poisson`/`gamma`/`constant`).
    pub rate_mode: Option<String>,
    /// Gamma smoothness shape.
    pub smoothness: Option<f64>,
    /// Concurrency-ramp duration, seconds.
    pub concurrency_ramp: Option<f64>,
    /// Rate-ramp duration, seconds.
    pub rate_ramp: Option<f64>,
    /// Post-send cancellation `(rate, delay)`.
    pub cancellation: Option<(f64, f64)>,
    /// User-centric arrival `(rate, users)`.
    pub user_centric: Option<(f64, u32)>,
    pub request_count: Option<u64>,
    pub benchmark_duration: Option<f64>,
    pub grace_period: Option<f64>,
    pub warmup: Option<Warmup>,
    /// Runtime worker count (`runtime.workers`; `None` = runner auto-selects).
    pub runtime_workers: Option<u32>,
    /// Adaptive-scaling minimum worker count (`runtime.workers_min`).
    pub runtime_workers_min: Option<u32>,
    /// Cellular (multi-process) cell count (`runtime.cells`; `1` = single).
    pub runtime_cells: u32,
    /// Admission strategy for `workers>1` scheduled execution (`runtime.dispatch`
    /// / `--dispatch`). `None` omits the wire field, decoded as `Global`.
    pub runtime_dispatch: Option<DispatchMode>,
    /// Explicit `--hop-routing` worker-assignment policy for the
    /// single-coordinator modes `global-hop`/`global-push` (`workers > 1`).
    /// `None` lets resolution derive it from the resolved
    /// connection-reuse strategy (`sticky` under `sticky-user-sessions`, else
    /// `round-robin`).
    pub runtime_hop_routing: Option<HopRouting>,
    pub random_seed: Option<u64>,
    /// Per-dataset sampling seed (`dataset.random_seed`). The `--random-seed`
    /// flag sets both this and `random_seed`; a YAML top-level `randomSeed` sets
    /// only `random_seed` (the run seed), so the two are tracked separately.
    pub dataset_random_seed: Option<u64>,
    /// File-backed dataset path (mutually exclusive with the synthetic path).
    pub input_file: Option<PathBuf>,
    /// Exact inline system-prompt source before startup resolution.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// File-backed system-prompt source before startup resolution.
    #[serde(default)]
    pub system_prompt_file: Option<PathBuf>,
    /// Recorded-agent replay policy for an `agent_recording` file dataset.
    pub recorded_agent_graph: Option<RecordedAgentGraphConfig>,
    /// Free-form endpoint hardware provenance.
    pub hardware_description: Option<String>,
    /// Endpoint placement relative to tool execution.
    pub endpoint_placement: String,
    /// Inline file-dataset records authored directly in the config (mutually
    /// exclusive with `input_file`; emitted verbatim as `records` on the wire).
    pub inline_records: Option<serde_json::Value>,
    /// Named submission scenario (`--scenario`; `cfg.scenario`).
    pub scenario: Option<String>,
    /// WEKA reconstruction semantics (`--weka-semantics`; legacy|graph-ir).
    pub weka_semantics: Option<String>,
    /// Ignore recorded trace inter-message/inter-request delays for graph-ir runs
    /// (`--ignore-trace-delays`): fire every node as soon as its inputs are ready.
    pub ignore_trace_delays: bool,
    /// Whether `--ignore-trace-delays` was explicitly set (distinguishes user
    /// intent from the default for the scenario guard).
    pub ignore_trace_delays_explicit: bool,
    /// Recorded-graph trajectory-start window lower ratio (`--trajectory-start-min-ratio`).
    pub trajectory_start_min_ratio: f64,
    /// Recorded-graph trajectory-start window upper ratio (`--trajectory-start-max-ratio`).
    pub trajectory_start_max_ratio: f64,
    /// Relax cross-field validation (`--unsafe-override`; `cfg.unsafe_override`).
    pub unsafe_override: bool,
    /// Agentic cache-warmup duration, seconds (auto-creates a warmup phase).
    pub agentic_cache_warmup_duration: Option<f64>,
    /// Rankings/rerank query-passage generation (present when a rankings flag is set).
    pub rankings: Option<crate::config::model::dataset::Rankings>,
    /// Accuracy-benchmark policy (present when `--accuracy-benchmark` is set).
    pub accuracy: Option<crate::config::model::config::Accuracy>,
    /// Recorded-graph synthesis block (present when a `--synthesis-*` flag is set).
    pub synthesis: Option<serde_json::Value>,
    /// Parsed public-dataset loader filters (`--dataset-filter key=value`).
    pub dataset_filters: Option<serde_json::Map<String, serde_json::Value>>,
    /// File dataset format id (`--custom-dataset-type`).
    pub custom_dataset_type: Option<String>,
    /// Named public dataset (mutually exclusive with synthetic/file).
    pub public_dataset: Option<String>,
    /// HuggingFace subset override for the public dataset.
    pub hf_subset: Option<String>,
    /// Arbitrary Hugging Face dataset repository ID (`--hf-dataset`); bypasses the catalog.
    pub hf_dataset: Option<String>,
    /// Hugging Face dataset split (`--hf-split`); auto-resolved if omitted.
    pub hf_split: Option<String>,
    /// Hugging Face dataset git revision (`--hf-revision`).
    pub hf_revision: Option<String>,
    /// Forced prompt column for `--hf-dataset` (`--hf-text-column`).
    pub hf_text_column: Option<String>,
    /// Forced completion/output column for `--hf-dataset` (`--hf-output-column`).
    pub hf_output_column: Option<String>,
    /// Fixed output length for `--hf-dataset` (`--hf-output-len`).
    pub hf_output_len: Option<u32>,
    /// Forced loader format for `--hf-dataset` (`--hf-format`); default `hf`.
    pub hf_format: Option<String>,
    /// Inter-turn delay cap, seconds (file datasets).
    pub inter_turn_delay_cap_seconds: Option<f64>,
    /// Fetch remote image URLs and inline them as data URLs at dataset
    /// generation (`--prefetch-media-urls`); file/public datasets only.
    pub prefetch_media_urls: bool,
    /// Strip repeated image content once observed within a session
    /// (`--uuid-and-strip`), single_turn only.
    pub uuid_and_strip: bool,
    /// `baseten_trace` replay-timing knobs.
    pub replay_speedup: Option<f64>,
    pub max_idle_gap_cap_seconds: Option<f64>,
    pub open_loop_replay: bool,
    pub open_loop_strict: bool,
    pub omit_kv_hints: bool,
    pub force_min_tokens: bool,
    /// Fixed-schedule replay (timestamp-driven); carries the auto-offset flag.
    pub fixed_schedule: Option<bool>,
    /// Fixed-schedule start/end offsets.
    pub fixed_schedule_start_offset: Option<i64>,
    pub fixed_schedule_end_offset: Option<i64>,
    /// Model-selection strategy override.
    pub model_strategy: Option<ModelStrategy>,
    /// Timeslice window, seconds.
    pub slice_duration: Option<f64>,
    /// Synthetic input-token block size.
    pub isl_block_size: Option<u32>,
    /// Fraction of synthetic prompts drawing the shared reusable prefix.
    pub prefix_reuse_fraction: Option<f64>,
    /// Fraction of a reusing prompt's input length occupied by the shared prefix.
    pub prefix_reuse_ratio: Option<f64>,
    /// Authored prompt corpus selector for synthesized prompt content.
    pub prompt_corpus: Option<String>,
    /// Bounded-memory sketch metric retention.
    pub sketch_metrics: bool,
    /// Closed-loop steady-state summary for concurrency-target runs.
    pub steady_state: bool,
    /// Steady-state occupancy fraction of the concurrency target.
    pub steady_state_fraction: Option<f64>,
    /// Hybrid steady-state latency mode (full-run latency, windowed throughput).
    pub steady_state_hybrid: bool,
    /// Explicit text batch size for file-backed `random_pool` inputs.
    pub random_pool_text_batch_size: Option<u32>,
    /// Explicit image batch size for file-backed `random_pool` inputs.
    pub random_pool_image_batch_size: Option<u32>,
    /// Explicit audio batch size for file-backed `random_pool` inputs.
    pub random_pool_audio_batch_size: Option<u32>,
    /// Explicit video batch size for file-backed `random_pool` inputs.
    pub random_pool_video_batch_size: Option<u32>,
    /// Synthetic image spec (present when any image flag is set).
    pub image_spec: Option<ImageSpec>,
    /// Synthetic audio spec.
    pub audio_spec: Option<AudioSpec>,
    /// Synthetic video spec.
    pub video_spec: Option<VideoSpec>,
    /// Adaptive-scale controller (present when --adaptive-scale is set).
    pub adaptive_scale: Option<AdaptiveScale>,
    /// Piecewise-linear request-rate schedule (mutually exclusive with scalar rate).
    pub request_rate_series: Option<RateSeries>,
    /// Shared-prefix / prefix-pool policy.
    pub prefix_prompts: Option<PrefixPrompts>,
    /// Dry-run dataset-analysis emission (present when `--dry-run` is set without
    /// `--no-dataset-analysis`).
    pub dataset_analysis: Option<DatasetAnalysisInputs>,
    /// Explicit ordered phase list from YAML (`phases:`); overrides warmup/profiling axes.
    pub phases_override: Option<Vec<Phase>>,
    pub artifact_dir: PathBuf,
}

/// Dry-run dataset-analysis knobs projected from the `--kv-*` /
/// `--dataset-analysis-*` flags into `artifacts.dataset_analysis_*`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DatasetAnalysisInputs {
    /// KV-cache block size (tokens) for the cache-reuse analysis.
    pub block_size: u32,
    /// Explicit realized-LRU capacity (blocks) sweep point, if requested.
    pub cache_blocks: Option<u64>,
    /// Emit per-conversation breakdowns.
    pub per_conversation: bool,
}

/// Baseten_trace replay knobs projected into loader `options`.
#[derive(Clone, Copy)]
struct BasetenReplayKnobs {
    inter_turn_delay_cap_seconds: Option<f64>,
    replay_speedup: Option<f64>,
    max_idle_gap_cap_seconds: Option<f64>,
    open_loop_replay: bool,
    open_loop_strict: bool,
    omit_kv_hints: bool,
    force_min_tokens: bool,
    trace_session_sample_ratio: Option<f64>,
    isl_block_size: Option<u32>,
}

impl BasetenReplayKnobs {
    fn from_inputs(inputs: &Inputs) -> Self {
        Self {
            inter_turn_delay_cap_seconds: inputs.inter_turn_delay_cap_seconds,
            replay_speedup: inputs.replay_speedup,
            max_idle_gap_cap_seconds: inputs.max_idle_gap_cap_seconds,
            open_loop_replay: inputs.open_loop_replay,
            open_loop_strict: inputs.open_loop_strict,
            omit_kv_hints: inputs.omit_kv_hints,
            force_min_tokens: inputs.force_min_tokens,
            trace_session_sample_ratio: inputs.trace_session_sample_ratio,
            isl_block_size: inputs.isl_block_size,
        }
    }

    fn insert_into(self, options: &mut serde_json::Map<String, serde_json::Value>) {
        if let Some(cap) = self.inter_turn_delay_cap_seconds {
            options.insert(
                "inter_turn_delay_cap_seconds".to_string(),
                serde_json::json!(cap),
            );
        }
        if let Some(speedup) = self.replay_speedup {
            options.insert("replay_speedup".to_string(), serde_json::json!(speedup));
        }
        if let Some(cap) = self.max_idle_gap_cap_seconds {
            options.insert(
                "max_idle_gap_cap_seconds".to_string(),
                serde_json::json!(cap),
            );
        }
        if !self.open_loop_replay {
            options.insert("open_loop_replay".to_string(), serde_json::json!(false));
        }
        if self.open_loop_strict {
            options.insert("open_loop_strict".to_string(), serde_json::json!(true));
        }
        if self.omit_kv_hints {
            options.insert("omit_kv_hints".to_string(), serde_json::json!(true));
        }
        if !self.force_min_tokens {
            options.insert("force_min_tokens".to_string(), serde_json::json!(false));
        }
        if let Some(ratio) = self.trace_session_sample_ratio {
            options.insert(
                "trace_session_sample_ratio".to_string(),
                serde_json::json!(ratio),
            );
        }
        if let Some(block_size) = self.isl_block_size {
            options.insert("block_size".to_string(), serde_json::json!(block_size));
        }
    }
}

/// Whether the resolved dataset uses the `baseten_trace` loader.
///
/// True for file datasets with `--custom-dataset-type baseten_trace`, for
/// `--hf-dataset` with `--hf-format baseten_trace`, and for catalog public
/// datasets whose format is `baseten_trace`.
fn is_baseten_trace_dataset(inputs: &Inputs) -> bool {
    if inputs.custom_dataset_type.as_deref() == Some("baseten_trace") {
        return true;
    }
    if inputs.hf_dataset.is_some() && inputs.hf_format.as_deref() == Some("baseten_trace") {
        return true;
    }
    inputs
        .public_dataset
        .as_deref()
        .and_then(crate::config::model::public_catalog::lookup)
        .is_some_and(|meta| meta.format == "baseten_trace")
}

/// Insert baseten_trace replay knobs into a loader `options` bag.
fn insert_baseten_replay_options(
    options: &mut serde_json::Map<String, serde_json::Value>,
    knobs: BasetenReplayKnobs,
) {
    knobs.insert_into(options);
}

/// Reject `extra`-input keys the baseten_trace loader injects per-turn.
///
/// Ports Python's `_reject_baseten_trace_extra_input_collisions`:
/// loader-injected per-turn values (`min_tokens` from the recorded output
/// length, `hash_ids`/`block_size` KV hints) overwrite endpoint-level
/// extras, so the user's value would be silently clobbered on the wire.
/// `max_tokens` is not guarded: user extras win over the loader for that key.
///
/// Operates on the shared [`Inputs`] bag (`inputs.extra`,
/// `inputs.custom_dataset_type`, `inputs.force_min_tokens`,
/// `inputs.omit_kv_hints`) so both the `--extra-inputs` flags path and the
/// YAML `endpoint.extra` path enforce it identically — it is the single source
/// of truth, called from `build()`.
fn validate_baseten_extra_input_collisions(inputs: &Inputs) -> anyhow::Result<()> {
    if !is_baseten_trace_dataset(inputs) {
        return Ok(());
    }
    let extra = &inputs.extra;
    let mut collisions: Vec<(&str, &str)> = Vec::new();
    if inputs.force_min_tokens && extra.contains_key("min_tokens") {
        collisions.push(("min_tokens", "--no-force-min-tokens"));
    }
    if !inputs.omit_kv_hints {
        for key in ["hash_ids", "block_size"] {
            if extra.contains_key(key) {
                collisions.push((key, "--omit-kv-hints"));
            }
        }
    }
    if collisions.is_empty() {
        return Ok(());
    }
    let message = collisions
        .iter()
        .map(|(key, flag)| {
            format!(
                "--extra-inputs {key} is overwritten per-turn by the baseten_trace loader; \
                 pass {flag} to send your value instead"
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    anyhow::bail!(message)
}

/// Reject baseten_trace-only replay knobs on incompatible datasets.
///
/// Ports Python's `_reject_baseten_only_trace_flags`: these knobs are only
/// consumed by the baseten_trace loader; on any other dataset they would
/// silently no-op, hiding user error. Scope cut from Python: Python's check
/// fires on any *explicit* mention of a flag (via `model_fields_set`), even
/// one matching its own default (e.g. an explicit `--force-min-tokens`
/// redundant with the default). Rust has no equivalent "was this flag
/// explicitly passed" signal on parsed bools without deeper clap
/// introspection, so this instead fires on a *non-default value* -- it
/// catches every case that would actually change behavior on the wrong
/// loader, just not a redundant explicit default.
/// Accepted when the dataset is file `baseten_trace`, a catalog public dataset
/// whose format is `baseten_trace`, or `--hf-dataset` with `--hf-format
/// baseten_trace`.
fn validate_baseten_only_trace_flags(inputs: &Inputs) -> anyhow::Result<()> {
    let mut set_flags = Vec::new();
    if inputs.replay_speedup.is_some() {
        set_flags.push("--replay-speedup");
    }
    if inputs.max_idle_gap_cap_seconds.is_some() {
        set_flags.push("--max-idle-gap-cap-seconds");
    }
    if !inputs.open_loop_replay {
        set_flags.push("--open-loop-replay/--no-open-loop-replay");
    }
    if inputs.open_loop_strict {
        set_flags.push("--open-loop-strict");
    }
    if inputs.omit_kv_hints {
        set_flags.push("--omit-kv-hints");
    }
    if !inputs.force_min_tokens {
        set_flags.push("--force-min-tokens/--no-force-min-tokens");
    }
    if inputs.trace_session_sample_ratio.is_some() {
        set_flags.push("--trace-session-sample-ratio");
    }
    if set_flags.is_empty() {
        return Ok(());
    }
    if is_baseten_trace_dataset(inputs) {
        return Ok(());
    }
    let msg = format!(
        "{} is only supported by the baseten_trace loader",
        set_flags.join(", ")
    );
    if let Some(format) = &inputs.custom_dataset_type {
        anyhow::bail!("{msg}, but --custom-dataset-type is {format}.");
    }
    if let Some(format) = &inputs.hf_format {
        anyhow::bail!("{msg}, but --hf-format is {format}.");
    }
    if let Some(name) = &inputs.public_dataset {
        let format = crate::config::model::public_catalog::lookup(name)
            .map(|meta| meta.format.as_str())
            .unwrap_or("unknown");
        anyhow::bail!("{msg}, but --public-dataset {name} uses format {format}.");
    }
    anyhow::bail!(
        "{msg}; provide --input-file with --custom-dataset-type baseten_trace, \
         a baseten_trace public dataset, or --hf-dataset with --hf-format baseten_trace."
    );
}

fn random_pool_batch_sizes(inputs: &Inputs) -> [(&'static str, Option<u32>); 4] {
    [
        ("text", inputs.random_pool_text_batch_size),
        ("image", inputs.random_pool_image_batch_size),
        ("audio", inputs.random_pool_audio_batch_size),
        ("video", inputs.random_pool_video_batch_size),
    ]
}

/// Reject file-random-pool controls on dataset paths that cannot preserve them.
fn validate_random_pool_batch_sizes(inputs: &Inputs) -> anyhow::Result<()> {
    let batch_sizes = random_pool_batch_sizes(inputs);
    if batch_sizes.iter().all(|(_, size)| size.is_none()) {
        return Ok(());
    }

    if inputs.hf_dataset.is_some() || inputs.public_dataset.is_some() {
        anyhow::bail!(
            "random_pool batch sizes cannot be used with a public dataset; use a file dataset with format random_pool"
        );
    }

    if inputs.input_file.is_none() && inputs.inline_records.is_none() {
        // The same authoring controls configure generated synthetic modalities.
        return Ok(());
    }

    let format = inputs
        .custom_dataset_type
        .as_deref()
        .unwrap_or("single_turn");
    anyhow::ensure!(
        format == "random_pool",
        "random_pool batch sizes require file dataset format random_pool, but the effective format is {format}"
    );

    if let Some(path) = inputs.input_file.as_ref().filter(|path| path.is_dir())
        && let Some((modality, size)) = batch_sizes.iter().find_map(|(modality, size)| {
            size.filter(|size| *size != 1).map(|size| (*modality, size))
        })
    {
        anyhow::bail!(
            "random_pool directory {} uses named pools, so {modality}_batch_size must be 1; got {size}",
            path.display()
        );
    }

    Ok(())
}

pub fn resolve(mut inputs: Inputs) -> anyhow::Result<BenchmarkRun> {
    let system_prompt = crate::config::system_prompt::resolve_system_prompt(
        inputs.system_prompt.as_deref(),
        inputs.system_prompt_file.as_deref(),
    )?;
    validate_baseten_only_trace_flags(&inputs)?;
    validate_baseten_extra_input_collisions(&inputs)?;
    validate_random_pool_batch_sizes(&inputs)?;
    if let Some(ratio) = inputs.trace_session_sample_ratio {
        anyhow::ensure!(
            ratio.is_finite() && ratio > 0.0 && ratio <= 1.0,
            "--trace-session-sample-ratio must be in (0.0, 1.0], got {ratio}"
        );
    }
    if let Some(threshold) = inputs.failed_request_threshold {
        anyhow::ensure!(
            threshold.is_finite() && (0.0..=1.0).contains(&threshold),
            "--failed-request-threshold must be in [0.0, 1.0], got {threshold}"
        );
    }
    if let Some(cap) = inputs.system_idle_gap_cap_seconds {
        anyhow::ensure!(
            cap.is_finite() && cap >= 0.0,
            "--system-idle-gap-cap-seconds must be finite and non-negative, got {cap}"
        );
    }
    if let Some(grace) = inputs.agentic_warmup_grace_period {
        anyhow::ensure!(
            grace.is_finite() && grace >= 0.0,
            "--agentic-warmup-grace-period must be finite and non-negative, got {grace}"
        );
    }
    // Effective weka semantics, resolved while `inputs` is still whole (needed
    // before scenario-lock materialization so a graph-ir-specific lock targets the
    // right arm).
    let weka_semantics = resolve_weka_semantics(&inputs);
    // Materialize the `--scenario` submission locks onto `inputs` so BOTH weka
    // runtimes honor them. The legacy path hardcodes `ignore_eos`/t*/cache-bust in
    // its lowering and never consults `inputs` for them; the graph-ir path instead
    // derives its composed wire body (from `inputs.extra`) and its t* warmup phase
    // (from `inputs.warmup` + the synthesis block) from `inputs`. Without this a
    // `--weka-semantics graph-ir` run under an agentic scenario silently drops
    // `ignore_eos` from the body and runs a single unprimed profiling phase.
    // Runs before `resolve_scenario_outcome` so the outcome report and its
    // conflict checks see the injected values.
    apply_scenario_graph_locks(&mut inputs, weka_semantics.as_deref())?;
    // Resolve legacy-AgentX scenario locks (`--scenario`) while `inputs` is still
    // whole (later lowering partially moves it). A hard scenario-lock conflict
    // fails resolution here.
    let scenario_outcome = resolve_scenario_outcome(&inputs)?;
    // Capture synthesis / cache-bust before later partial moves of `inputs`.
    let synthesis = resolve_synthesis(&inputs);
    let cache_bust = cache_bust_from_inputs(&inputs);
    // The agentic_replay (legacy weka) timing mode is a single coherent driver:
    // one workload instance owns the per-tree join gate, session-tree registry,
    // and recycle cursor. It runs global-dispatch, single-worker, non-cellular.
    // Cellular execution (`--cells > 1`) would partition a trajectory tree's root
    // and subagent children across cell processes, breaking join gating — reject
    // it with a clear error (use `--weka-semantics graph-ir` for cellular weka).
    if inputs.system_idle_gap_cap_seconds.is_some()
        && !matches!(
            weka_semantics.as_deref(),
            Some("legacy") | Some("agentx") | Some("graph-ir") | Some("graphir") | Some("graph_ir")
        )
    {
        anyhow::bail!(
            "--system-idle-gap-cap-seconds requires a Weka replay mode \
             (set --weka-semantics legacy or graph-ir)"
        );
    }
    if matches!(weka_semantics.as_deref(), Some("legacy") | Some("agentx")) {
        if inputs.runtime_cells > 1 {
            anyhow::bail!(
                "the agentic_replay (legacy weka) timing mode does not support cellular \
                 execution (--cells {}); it runs non-cellular. Use --weka-semantics \
                 graph-ir for cellular weka replay.",
                inputs.runtime_cells
            );
        }
        // agentic_replay mirrors Python's `1 strategy : 1 router : N workers`: ONE
        // central driver computes the whole dispatch schedule (t*, warmup, profiling
        // offsets, recycle, join-gating) and issues each request to a shared worker
        // pool. `global-hop` is that model in the Rust runtime — one coordinator
        // scheduling loop hops each request round-robin to a pool of worker transport
        // threads. `sharded`/`global` instead run N independent per-worker pipelines
        // (each with its own conversation partition + metrics slot space), which
        // splits trajectory trees and collides slots. Force `global-hop`; reject an
        // explicit conflicting mode so the choice is visible.
        match inputs.runtime_dispatch {
            None | Some(DispatchMode::GlobalHop) => {
                inputs.runtime_dispatch = Some(DispatchMode::GlobalHop);
            }
            Some(other) => anyhow::bail!(
                "the agentic_replay (legacy weka) timing mode requires --dispatch global-hop \
                 (a single central issuer over a shared worker pool); got {other:?}. Omit \
                 --dispatch (it defaults to global-hop here) or pass --dispatch global-hop."
            ),
        }
    }
    // Restrict `--agentic-cache-warmup-duration` to a weka reconstruction run.
    // Both weka arms consume the accelerated cache-warmup substage: the legacy
    // arm through `lower_legacy_agentic` (which recovers the value from the
    // authored phases and threads it onto its synthesized warmup barrier), and
    // the graph-ir arm through `build_pressure_recycle` in
    // `graph_phase_runtime`. Outside weka the value reaches no consumer and is
    // silently dropped, so an unguarded flag there is an invisible no-op; reject
    // it instead (ports Python's `validate_agentic_cache_warmup`).
    if inputs.agentic_cache_warmup_duration.is_some() && weka_semantics.is_none() {
        anyhow::bail!(
            "--agentic-cache-warmup-duration requires a weka reconstruction run \
             (set by --scenario inferencex-agentx-mvp or --weka-semantics \
             legacy|graph-ir); this run resolves to neither weka arm."
        );
    }
    let loadgen_overlay = crate::config::phase_validate::LoadgenOverlay::from_inputs(&inputs);
    if let Some(ref mut phases) = inputs.phases_override {
        apply_cli_loadgen_overlays(phases, &loadgen_overlay)?;
    }
    let primary_model = inputs.model_names[0].clone();

    let models = Models {
        strategy: inputs.model_strategy.unwrap_or(ModelStrategy::RoundRobin),
        items: inputs
            .model_names
            .iter()
            .map(|name| ModelItem {
                name: name.clone(),
                weight: None,
            })
            .collect(),
    };

    let endpoint_type_for_dataset_validation = inputs.endpoint_type.clone();
    let baseten_knobs = BasetenReplayKnobs::from_inputs(&inputs);
    // Resolve the effective connection-reuse strategy once so the hop-routing
    // default can derive from it (see `resolved_hop_routing`).
    let resolved_connection_reuse = inputs.connection_reuse.unwrap_or(ConnectionReuse::Pooled);
    // Effective single-coordinator worker-assignment policy: an explicit
    // `--hop-routing`/`runtime.hop_routing` always wins; absent, sticky
    // per-session connection reuse makes `Sticky` the sensible default (one
    // worker per session keeps the sticky pool warm), otherwise `RoundRobin`.
    // Inert unless the run is `global-hop`/`global-push` with `workers > 1`.
    let resolved_hop_routing =
        resolve_hop_routing(inputs.runtime_hop_routing, resolved_connection_reuse);
    let endpoint = Endpoint {
        urls: inputs.urls.iter().map(|u| normalize_url(u)).collect(),
        endpoint_type: EndpointType(inputs.endpoint_type),
        streaming: inputs.streaming,
        use_legacy_max_tokens: inputs.use_legacy_max_tokens,
        use_server_token_count: inputs.use_server_token_count,
        timeout_seconds: inputs.timeout_seconds.unwrap_or(DEFAULT_TIMEOUT_SECONDS),
        connection_reuse: resolved_connection_reuse,
        ssl_verify: inputs.ssl_verify.unwrap_or(true),
        uds_path: inputs.uds_path.clone(),
        connection_limit: DEFAULT_CONNECTION_LIMIT,
        keepalive_timeout: DEFAULT_KEEPALIVE_TIMEOUT,
        download_video_content: inputs.download_video_content,
        extra: inputs.extra.clone(),
        headers: inputs.headers,
        http2: false,
        wait_for_model_timeout: inputs.wait_for_model_timeout.unwrap_or(0.0),
        wait_for_model_interval: inputs
            .wait_for_model_interval
            .unwrap_or(DEFAULT_WAIT_FOR_MODEL_INTERVAL),
        wait_for_model_mode: inputs
            .wait_for_model_mode
            .unwrap_or(WaitForModelMode::Inference),
        path: inputs.endpoint_path,
        api_key: inputs.api_key,
        session_header: inputs.session_header,
        request_content_type: inputs.request_content_type,
        template: None,
        response_field: None,
        reset_kv_cache: inputs.reset_kv_cache,
        server_profiler: inputs.server_profiler,
        proxy: inputs.proxy,
        proxy_from_env: inputs.proxy_from_env,
    };

    // Placeholder-only model sets use the offline builtin tokenizer unless overridden.
    let tokenizer_name = inputs.tokenizer_name.clone().unwrap_or_else(|| {
        if inputs.model_names.iter().all(|m| is_fake_model_name(m)) {
            "builtin".to_string()
        } else {
            primary_model.clone()
        }
    });
    let tokenizer = Tokenizer {
        name: tokenizer_name,
        revision: inputs
            .tokenizer_revision
            .unwrap_or_else(|| "main".to_string()),
        trust_remote_code: inputs.tokenizer_trust,
        apply_chat_template: inputs.apply_chat_template,
        server_url: inputs.server_tokenizer_url.clone(),
    };

    let dataset = if let Some(id) = &inputs.hf_dataset {
        anyhow::ensure!(
            inputs.public_dataset.is_none(),
            "--hf-dataset and --public-dataset are mutually exclusive"
        );
        let mut options = inputs.dataset_filters.clone().unwrap_or_default();
        if let Some(c) = &inputs.hf_text_column {
            options.insert("text_column".to_string(), serde_json::json!(c));
        }
        if let Some(c) = &inputs.hf_output_column {
            options.insert("output_column".to_string(), serde_json::json!(c));
        }
        if let Some(n) = inputs.hf_output_len {
            options.insert("output_len".to_string(), serde_json::json!(n));
        }
        let format = inputs.hf_format.clone().unwrap_or_else(|| "hf".to_string());
        if format == "baseten_trace" {
            insert_baseten_replay_options(&mut options, baseten_knobs);
        }
        let mut source = serde_json::Map::new();
        source.insert("type".to_string(), serde_json::json!("hugging_face"));
        source.insert("dataset".to_string(), serde_json::json!(id));
        source.insert(
            "subset".to_string(),
            serde_json::json!(inputs.hf_subset.clone().unwrap_or_default()),
        );
        source.insert(
            "split".to_string(),
            serde_json::json!(inputs.hf_split.clone().unwrap_or_default()),
        );
        if let Some(rev) = &inputs.hf_revision {
            source.insert("revision".to_string(), serde_json::json!(rev));
        }
        Dataset::Public(crate::config::model::dataset::PublicDataset {
            system_prompt: system_prompt.clone(),
            cache_bust,
            name: id.clone(),
            format,
            source: serde_json::Value::Object(source),
            options,
            sampling: Sampling(inputs.sampling.clone()),
            entries: inputs.dataset_entries,
            random_seed: inputs.dataset_random_seed,
            prompts: authored_prompt_selection(inputs.prompt_corpus.as_deref()),
            synthesis: synthesis.clone(),
            prefetch_media_urls: inputs.prefetch_media_urls,
        })
    } else if inputs.public_dataset.as_deref() == Some("weka_hf") {
        let repo = inputs.hf_weka_dataset.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--hf-weka-dataset is required with --public-dataset weka_hf")
        })?;
        let mut source = serde_json::Map::new();
        source.insert("type".to_string(), serde_json::json!("hugging_face"));
        source.insert("dataset".to_string(), serde_json::json!(repo));
        source.insert("subset".to_string(), serde_json::json!("default"));
        source.insert(
            "split".to_string(),
            serde_json::json!(inputs.hf_split.clone().unwrap_or_else(|| "train".into())),
        );
        if let Some(rev) = &inputs.hf_revision {
            source.insert("revision".to_string(), serde_json::json!(rev));
        }
        let options = inputs.dataset_filters.clone().unwrap_or_default();
        if let Some(subset) = &inputs.hf_subset {
            source.insert("subset".to_string(), serde_json::json!(subset));
        }
        Dataset::Public(crate::config::model::dataset::PublicDataset {
            system_prompt: system_prompt.clone(),
            cache_bust,
            name: "weka_hf".to_string(),
            format: "weka_trace".to_string(),
            source: serde_json::Value::Object(source),
            options,
            sampling: Sampling(inputs.sampling.clone()),
            entries: inputs.dataset_entries,
            random_seed: inputs.dataset_random_seed,
            prompts: authored_prompt_selection(inputs.prompt_corpus.as_deref()),
            synthesis: synthesis.clone(),
            prefetch_media_urls: inputs.prefetch_media_urls,
        })
    } else if let Some(name) = &inputs.public_dataset {
        let meta = crate::config::model::public_catalog::lookup(name)
            .ok_or_else(|| anyhow::anyhow!("unknown public dataset {name:?}"))?;
        let mut options = meta.options.clone();
        // Explicit filters override catalog loader options.
        if let Some(filters) = &inputs.dataset_filters {
            for (k, v) in filters {
                options.insert(k.clone(), v.clone());
            }
        }
        // Exgentic loaders require the fixed-schedule decision in their options.
        if matches!(meta.format.as_str(), "exgentic" | "exgentic_v2") {
            options.insert(
                "fixed_schedule".to_string(),
                serde_json::json!(inputs.fixed_schedule.is_some()),
            );
        }
        // Cap the loaded corpus only when an entry count was set explicitly
        // (`inputs.dataset_entries`, `None` when unset). The `DEFAULT_ENTRIES`
        // fallback in `inputs.entries` is a *synthetic* default (how many prompts
        // to generate); applying it here silently truncated a recorded/public
        // corpus to 100. Unset now means "load the full dataset" — the loader
        // caps at whatever the corpus actually holds. Matches the HF-direct path
        // above, which already uses `inputs.dataset_entries`.
        if let Some(max) = crate::config::model::public_catalog::max_conversations(
            meta,
            inputs.dataset_entries,
            inputs.request_count,
        ) {
            options.insert("max_conversations".to_string(), serde_json::json!(max));
        }
        if meta.format == "baseten_trace" {
            insert_baseten_replay_options(&mut options, baseten_knobs);
        }
        let mut source = meta.source.clone();
        if let (Some(subset), Some(obj)) = (&inputs.hf_subset, source.as_object_mut()) {
            obj.insert("subset".to_string(), serde_json::json!(subset));
        }
        Dataset::Public(crate::config::model::dataset::PublicDataset {
            system_prompt: system_prompt.clone(),
            cache_bust,
            name: name.clone(),
            format: meta.format.clone(),
            source,
            options,
            sampling: Sampling(inputs.sampling.clone()),
            entries: inputs.dataset_entries,
            random_seed: inputs.dataset_random_seed,
            prompts: authored_prompt_selection(inputs.prompt_corpus.as_deref()),
            synthesis: synthesis.clone(),
            prefetch_media_urls: inputs.prefetch_media_urls,
        })
    } else if inputs.input_file.is_some() || inputs.inline_records.is_some() {
        if inputs.uuid_and_strip && endpoint_type_for_dataset_validation != "chat" {
            anyhow::bail!("--uuid-and-strip requires endpoint type 'chat'");
        }
        Dataset::File(crate::config::model::dataset::FileDataset {
            system_prompt: system_prompt.clone(),
            cache_bust,
            // Path-backed inputs are auto-detected; inline records require a format.
            format: inputs.custom_dataset_type.clone().or_else(|| {
                inputs
                    .inline_records
                    .is_some()
                    .then(|| "single_turn".to_string())
            }),
            sampling: Sampling(inputs.sampling.clone()),
            options: {
                let mut o = serde_json::Map::new();
                // Trace formats define different default KV block sizes.
                let format = inputs
                    .custom_dataset_type
                    .as_deref()
                    .unwrap_or("single_turn");
                if format == "random_pool" {
                    for (name, batch_size) in [
                        ("text_batch_size", inputs.random_pool_text_batch_size),
                        ("image_batch_size", inputs.random_pool_image_batch_size),
                        ("audio_batch_size", inputs.random_pool_audio_batch_size),
                        ("video_batch_size", inputs.random_pool_video_batch_size),
                    ] {
                        if let Some(batch_size) = batch_size {
                            o.insert(name.to_string(), serde_json::json!(batch_size));
                        }
                    }
                }
                match format {
                    "mooncake_trace" => {
                        o.insert("block_size".to_string(), serde_json::json!(512));
                    }
                    "bailian_trace" => {
                        o.insert("block_size".to_string(), serde_json::json!(16));
                    }
                    "tracelab" => {
                        o.insert("block_size".to_string(), serde_json::json!(64));
                    }
                    _ => {}
                }
                // Shared across DAG JSONL and baseten closed-loop think-times.
                if let Some(cap) = inputs.inter_turn_delay_cap_seconds {
                    o.insert(
                        "inter_turn_delay_cap_seconds".to_string(),
                        serde_json::json!(cap),
                    );
                }
                if inputs.uuid_and_strip {
                    o.insert("uuid_and_strip".to_string(), serde_json::json!(true));
                }
                if format == "baseten_trace" {
                    // Re-run insert after the shared inter-turn cap so baseten
                    // knobs (including a duplicate inter-turn write) stay in one
                    // helper shared with the public/hf paths.
                    insert_baseten_replay_options(&mut o, baseten_knobs);
                } else if let Some(block_size) = inputs.isl_block_size {
                    o.insert("block_size".to_string(), serde_json::json!(block_size));
                }
                o
            },
            // Path-backed and inline-record datasets are mutually exclusive.
            path: inputs.input_file.as_ref().map(|p| absolute_path(p)),
            // Fixed-schedule derives the count into the phase's `requests`, not
            // the dataset's `entries`; otherwise the explicit count (if any).
            entries: if inputs.fixed_schedule.is_some() {
                None
            } else {
                inputs.dataset_entries
            },
            random_seed: inputs.dataset_random_seed,
            osl: inputs.osl.clone(),
            prompts: authored_prompt_selection(inputs.prompt_corpus.as_deref()),
            records: inputs.inline_records.clone(),
            synthesis,
            graph: inputs.recorded_agent_graph.clone(),
            prefetch_media_urls: inputs.prefetch_media_urls,
        })
    } else {
        Dataset::Synthetic(Synthetic {
            system_prompt,
            prompts: Prompts {
                cache_bust,
                batch_size: inputs.batch_size,
                isl: inputs.isl.clone(),
                osl: inputs.osl.clone(),
                num_prefix_prompts: None,
                prefix_prompt_length: None,
                block_size: inputs.isl_block_size,
                corpus: inputs.prompt_corpus.clone(),
                sequence_distribution: inputs.sequence_distribution.clone(),
                prefix_reuse_fraction: inputs.prefix_reuse_fraction,
                prefix_reuse_ratio: inputs.prefix_reuse_ratio,
            },
            prefix_prompts: inputs.prefix_prompts.clone(),
            images: inputs.image_spec.clone(),
            audio: inputs.audio_spec.clone(),
            video: inputs.video_spec.clone(),
            rankings: inputs.rankings.clone(),
            sampling: Sampling(inputs.sampling.clone()),
            turns: inputs.turns.clone(),
            turn_delay_ratio: inputs.turn_delay_ratio,
            entries: Some(inputs.entries),
            random_seed: inputs.dataset_random_seed,
            num_conversations: None,
            turn_delay_ms: inputs.turn_delay_ms.clone(),
        })
    };

    // An unbounded run (no count, duration, schedule, user-centric mode, or
    // sessions bound) defaults to a fixed request count. A session-bounded run
    // (`--num-conversations` / a `sessions:` phase) is session-bounded only:
    // stamping a defaulted `requests` alongside `sessions` would over-constrain
    // the phase (and trips the cellular graph gate, which rejects a static
    // `requests` budget on a sessions-partitioned run).
    let effective_requests = inputs.request_count.or_else(|| {
        (inputs.benchmark_duration.is_none()
            && inputs.fixed_schedule.is_none()
            && inputs.user_centric.is_none()
            && inputs.sessions.is_none())
        .then_some(DEFAULT_REQUEST_COUNT)
    });
    let profiling_grace_period = inputs.grace_period.or_else(|| {
        inputs
            .benchmark_duration
            .map(|_| DEFAULT_BENCHMARK_GRACE_PERIOD_SECONDS)
    });
    let profiling = if let Some((rate, users)) = inputs.user_centric {
        Phase {
            common: PhaseCommon {
                timing_mode: None,
                name: "profiling".to_string(),
                kind: Some(PhaseRole::Profiling),
                exclude_from_results: false,
                seamless: false,
                requests: inputs.request_count,
                sessions: inputs.sessions,
                duration: inputs.benchmark_duration,
                prefill_concurrency: None,
                grace_period: profiling_grace_period,
                concurrency_ramp: None,
                prefill_ramp: None,
                rate_ramp: None,
                cancellation: None,
                agentic_cache_warmup_duration: None,
                agentic_warmup_grace_period: inputs.agentic_warmup_grace_period,
                failed_request_threshold: inputs.failed_request_threshold,
                adaptive_scale: None,
                rate_series: None,
            },
            kind: PhaseKind::UserCentric {
                rate,
                users,
                concurrency: inputs.concurrency,
            },
        }
    } else if let Some(auto_offset) = inputs.fixed_schedule {
        Phase {
            common: PhaseCommon {
                timing_mode: None,
                name: "profiling".to_string(),
                kind: Some(PhaseRole::Profiling),
                exclude_from_results: false,
                seamless: false,
                requests: inputs.request_count,
                sessions: None,
                duration: None,
                prefill_concurrency: None,
                grace_period: None,
                concurrency_ramp: None,
                prefill_ramp: None,
                rate_ramp: None,
                cancellation: None,
                agentic_cache_warmup_duration: None,
                agentic_warmup_grace_period: inputs.agentic_warmup_grace_period,
                failed_request_threshold: inputs.failed_request_threshold,
                adaptive_scale: None,
                rate_series: None,
            },
            kind: PhaseKind::FixedSchedule {
                auto_offset,
                start_offset: inputs.fixed_schedule_start_offset,
                end_offset: inputs.fixed_schedule_end_offset,
            },
        }
    } else {
        // `--request-rate-series` alone selects a rate-controlled phase whose
        // bootstrap QPS is the series' first point (mutually exclusive with
        // `--request-rate` at flag resolve time).
        let effective_rate = inputs.request_rate.or_else(|| {
            inputs
                .request_rate_series
                .as_ref()
                .map(|series| series.initial_qps())
        });
        let mut phase = build_phase(
            "profiling",
            false,
            inputs.concurrency.unwrap_or(1),
            effective_rate,
            inputs
                .rate_mode
                .as_deref()
                .or_else(|| inputs.request_rate_series.is_some().then_some("constant")),
            inputs.smoothness,
            inputs.concurrency,
            effective_requests,
            inputs.sessions,
            inputs.benchmark_duration,
            profiling_grace_period,
        );
        phase.common.adaptive_scale = inputs.adaptive_scale.clone();
        phase.common.prefill_concurrency = inputs.prefill_concurrency;
        phase.common.prefill_ramp = inputs.prefill_ramp.map(linear_ramp);
        phase.common.concurrency_ramp = inputs.concurrency_ramp.map(linear_ramp);
        phase.common.rate_ramp = inputs.rate_ramp.map(linear_ramp);
        phase.common.cancellation = inputs
            .cancellation
            .map(|(rate, delay)| crate::config::model::phase::Cancellation { rate, delay });
        phase.common.rate_series = inputs.request_rate_series.clone();
        phase.common.agentic_warmup_grace_period = inputs.agentic_warmup_grace_period;
        phase.common.failed_request_threshold = inputs.failed_request_threshold;
        phase
    };
    let mut phases = match inputs.phases_override.take() {
        Some(authored) => authored,
        None => {
            let mut phases = Vec::new();
            if let Some(warmup) = inputs.warmup {
                let concurrency = warmup.concurrency.or(inputs.concurrency);
                let mut wp = build_phase(
                    "warmup",
                    true,
                    concurrency.unwrap_or(1),
                    warmup.rate,
                    warmup.rate_mode.as_deref(),
                    None,
                    concurrency,
                    warmup.requests,
                    warmup.sessions,
                    warmup.duration,
                    warmup.grace_period,
                );
                wp.common.prefill_concurrency = warmup.prefill_concurrency;
                wp.common.concurrency_ramp = warmup.concurrency_ramp.map(linear_ramp);
                wp.common.rate_ramp = warmup.rate_ramp.map(linear_ramp);
                wp.common.prefill_ramp = warmup.prefill_ramp.map(linear_ramp);
                wp.common.agentic_cache_warmup_duration = inputs.agentic_cache_warmup_duration;
                phases.push(wp);
            } else if let Some(dur) = inputs.agentic_cache_warmup_duration {
                let mut wp = build_phase(
                    "warmup", true, 1, None, None, None, None, None, None, None, None,
                );
                wp.common.agentic_cache_warmup_duration = Some(dur);
                phases.push(wp);
            }
            phases.push(profiling);
            phases
        }
    };
    for phase in &mut phases {
        if phase.common.exclude_from_results {
            continue;
        }
        if phase.common.agentic_warmup_grace_period.is_none() {
            phase.common.agentic_warmup_grace_period = inputs.agentic_warmup_grace_period;
        }
        if phase.common.failed_request_threshold.is_none() {
            phase.common.failed_request_threshold = inputs.failed_request_threshold;
        }
    }
    normalize_and_validate_phases(&mut phases)?;

    let endpoint_type = endpoint.endpoint_type.0.clone();
    let endpoint_urls = endpoint.urls.clone();
    let is_dynosim = inputs.transport.is_dynosim();
    // Both DynoSim and the dry-run fake leaf open no sockets, so every online
    // sidecar (GPU telemetry, server-metrics scraping, network-latency probing)
    // is forced off for either. Per-record artifacts, however, stay available for
    // dry-run (it fabricates a real RequestRecord), so those key off `is_dynosim`.
    let no_server_sidecars = is_dynosim || inputs.transport.is_dry_run();
    // The environment and CLI independently enable sketch retention.
    let sketch_metrics = inputs.sketch_metrics || env_sketch_enabled();
    // DynoSim forces all sidecars off; otherwise GPU-telemetry and
    // server-metrics scraping are enabled by default and independently toggled.
    let gpu_enabled = inputs.gpu_telemetry_enabled && !no_server_sidecars;
    let server_enabled = inputs.server_metrics_enabled && !no_server_sidecars;
    let default_gpu_cfg = crate::config::model::telemetry::GpuTelemetryConfig::default();
    // A fixed mean takes precedence over active network probing.
    let mut network_latency_cfg = crate::config::model::telemetry::NetworkLatencyConfig::default();
    let network_latency_sidecar = if no_server_sidecars {
        None
    } else if let Some(mean_ms) = inputs.network_latency_mean {
        network_latency_cfg.enabled = true;
        network_latency_cfg.mean_ms = Some(mean_ms);
        Some(crate::config::model::telemetry::NetworkLatencySidecar::fixed(mean_ms))
    } else if let Some(ping) = inputs.network_latency_probe {
        network_latency_cfg.enabled = true;
        network_latency_cfg.ping_interval = ping;
        Some(crate::config::model::telemetry::NetworkLatencySidecar::probe(ping))
    } else {
        None
    };
    // The raw policy is the single source of the collector selection; the sidecar
    // is lowered from it so the two cannot describe different collectors. It is
    // built (and therefore validated) even when telemetry is off, so an unusable
    // selection fails at resolve time rather than the next time it is enabled.
    let gpu_cfg = crate::config::model::telemetry::GpuTelemetryConfig {
        enabled: inputs.gpu_telemetry_enabled,
        collector: inputs
            .gpu_telemetry_collector
            .clone()
            .unwrap_or_else(|| default_gpu_cfg.collector.clone()),
        mode: default_gpu_cfg.mode.clone(),
        metrics_file: inputs.gpu_telemetry_metrics_file.clone(),
        urls: inputs.gpu_telemetry_urls.clone(),
    };
    let mut gpu_sidecar =
        crate::config::model::telemetry::GpuTelemetrySidecar::from_config(&gpu_cfg)?;
    if inputs.profile_export_prefix.is_some() {
        let stem = artifact_export_stem(inputs.profile_export_prefix.as_deref());
        gpu_sidecar.records_path = format!("{stem}_gpu_telemetry.jsonl");
    }
    let sidecars = crate::config::model::telemetry::Sidecars {
        gpu_telemetry: gpu_enabled.then_some(gpu_sidecar),
        server_metrics: server_enabled.then(|| {
            let mut all_urls = endpoint_urls.clone();
            all_urls.extend(inputs.server_metrics_urls.iter().cloned());
            let sc = crate::config::model::telemetry::ServerMetricsSidecar::from_endpoint_urls(
                &all_urls,
            );
            match &inputs.server_metrics_formats {
                Some(formats) => sc.with_formats(formats.clone()),
                None => sc,
            }
        }),
        network_latency: network_latency_sidecar,
        // Opt-in run-owned content server, enabled purely through the
        // `AIPERF_CONTENT_SERVER_*` environment (no dedicated flag): when set it
        // externalizes generated images/videos as HTTP URLs. The runtime enforces
        // its online-HTTP placement rules and directory validation.
        content_server: crate::config::model::telemetry::ContentServerSidecar::from_env(),
    };
    let server_cfg = crate::config::model::telemetry::ServerMetricsConfig {
        enabled: inputs.server_metrics_enabled,
        // Config preserves authored URLs; sidecar construction normalizes them.
        urls: inputs.server_metrics_urls.clone(),
        formats: inputs.server_metrics_formats.clone().unwrap_or_else(|| {
            crate::config::model::telemetry::ServerMetricsConfig::default().formats
        }),
        ..Default::default()
    };
    let is_recorded_agent_replay = inputs.custom_dataset_type.as_deref() == Some("agent_recording");
    let recorded_agent_executes_tools = inputs
        .recorded_agent_graph
        .as_ref()
        .is_some_and(|graph| graph.execute_tools);
    let mut cfg = BenchmarkConfig {
        models: Some(models),
        endpoint: Some(endpoint),
        tokenizer: Some(tokenizer),
        transport: Some(inputs.transport),
        runtime: Some(Runtime {
            workers: inputs.runtime_workers,
            workers_min: inputs.runtime_workers_min,
            workers_max: None,
            cells: inputs.runtime_cells,
            dispatch: inputs.runtime_dispatch,
            hop_routing: Some(resolved_hop_routing),
        }),
        metrics: Some(Metrics {
            slos: inputs.slos.clone(),
            slice_duration_seconds: inputs.slice_duration,
            sketch: sketch_metrics.then_some(true),
            steady_state: inputs.steady_state.then_some(
                crate::config::model::metrics::SteadyState {
                    enabled: true,
                    fraction: inputs.steady_state_fraction,
                    hybrid_latency: inputs.steady_state_hybrid,
                },
            ),
        }),
        slos: (!inputs.slos.is_empty()).then(|| inputs.slos.clone()),
        artifacts: Some({
            // Sketch retention and DynoSim do not provide per-record values.
            let per_record = !sketch_metrics && !is_dynosim;
            let has = |f: &str| inputs.records_formats.iter().any(|x| x == f);
            let stem = artifact_export_stem(inputs.profile_export_prefix.as_deref());
            Artifacts {
                trace: (inputs.export_trace || inputs.show_trace_timing) && per_record,
                inputs_path: "inputs.json".to_string(),
                // `raw` forces the base JSONL on even when the format list omits it.
                records_path: (per_record && (has("jsonl") || inputs.export_raw))
                    .then(|| format!("{stem}.jsonl")),
                records_csv_path: (per_record && has("csv")).then(|| format!("{stem}_records.csv")),
                records_parquet_path: (per_record && has("parquet"))
                    .then(|| format!("{stem}.parquet")),
                raw_path: (per_record && inputs.export_raw).then(|| format!("{stem}_raw.jsonl")),
                outputs_path: (per_record && inputs.export_outputs_json)
                    .then(|| "outputs.json".to_string()),
                // Dry-run dataset analysis: emit beside this run-relative base path.
                // The runtime writes `dataset_analysis.{txt,json,csv,html}` next to it.
                dataset_analysis_path: inputs
                    .dataset_analysis
                    .as_ref()
                    .map(|_| "dataset_analysis.json".to_string()),
                dataset_analysis_block_size: inputs.dataset_analysis.as_ref().map(|a| a.block_size),
                dataset_analysis_cache_blocks: inputs
                    .dataset_analysis
                    .as_ref()
                    .and_then(|a| a.cache_blocks),
                user_files: (!inputs.user_files.is_empty()).then(|| inputs.user_files.clone()),
                dataset_analysis_per_conversation: inputs
                    .dataset_analysis
                    .as_ref()
                    .is_some_and(|a| a.per_conversation),
                graph_tool_time_path: (is_recorded_agent_replay && recorded_agent_executes_tools)
                    .then(|| "profile_export_graph_tool_time.json".to_string()),
                graph_trace_summary_path: is_recorded_agent_replay
                    .then(|| "profile_export_graph_trace_summary.json".to_string()),
                graph_replay_metrics_path: is_recorded_agent_replay
                    .then(|| "metrics.json".to_string()),
                graph_replay_metrics_csv_path: None,
                graph_replay_failures_path: is_recorded_agent_replay
                    .then(|| "failures.tsv".to_string()),
                graph_replay_provenance_path: is_recorded_agent_replay
                    .then(|| "replay-provenance.json".to_string()),
                graph_replay_backend_metadata_path: is_recorded_agent_replay
                    .then(|| "backend-metadata.json".to_string()),
            }
        }),
        metadata: (inputs.hardware_description.is_some() || inputs.endpoint_placement != "unknown")
            .then(|| Metadata {
                hardware: inputs.hardware_description.clone(),
                endpoint_placement: inputs.endpoint_placement.clone(),
            }),
        datasets: Some(vec![dataset]),
        phases: Some(phases),
        export: None,
        gpu_telemetry: Some(gpu_cfg),
        server_metrics: Some(server_cfg),
        network_latency: Some(network_latency_cfg),
        sidecars: Some(sidecars),
        accuracy: inputs.accuracy.clone(),
        endpoint_profiles: serde_json::Map::new(),
        failure_policy: None,
        scenario: inputs.scenario.clone(),
        weka_semantics,
        system_idle_gap_cap_seconds: inputs.system_idle_gap_cap_seconds,
        ignore_trace_delays: inputs.ignore_trace_delays,
        trajectory_start_max_ratio: inputs.trajectory_start_max_ratio,
        trajectory_start_min_ratio: inputs.trajectory_start_min_ratio,
        unsafe_override: inputs.unsafe_override,
    };

    let benchmark_id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
    // The genai-perf-v1 envelope requires an echoed config projection.
    let mut input_config = serde_json::to_value(&cfg).unwrap_or(serde_json::Value::Null);
    // Redaction applies only to the export copy, not runtime authentication.
    crate::config::redact::redact_input_config(&mut input_config);
    let mut export = crate::config::model::export::Export::build(
        &endpoint_type,
        &inputs.summary_formats,
        &benchmark_id,
        input_config.clone(),
        serde_json::json!({}),
        &inputs.model_names,
    );
    // The per-record OTLP sink is unavailable under sketch retention.
    if let Some(url) = &inputs.otel_url
        && !sketch_metrics
    {
        export.otel = Some(crate::config::model::export::OtelExport::build(
            url,
            &benchmark_id,
            &endpoint_type,
            &primary_model,
            inputs.otel_provider.as_deref(),
            &inputs.otel_resource_attributes,
        ));
    }
    export.mlflow =
        crate::config::model::export::MlflowExport::build(&inputs.mlflow, &benchmark_id);
    export.wandb = crate::config::model::export::WandbExport::build(&inputs.wandb, &benchmark_id);
    // Export formats use server-metrics config before export insertion.
    let sm_formats = cfg
        .server_metrics
        .as_ref()
        .map(|s| s.formats.clone())
        .unwrap_or_default();
    // `server_metrics_export.json` echoes the same config object, so it must reuse the
    // already-redacted projection; re-serializing `cfg` here would republish the
    // runtime credentials that the copy above exists to strip.
    export.server_metrics = crate::config::model::export::ServerMetricsExport::build(
        &sm_formats,
        server_enabled,
        crate::config::model::export::AIPERF_V1_VERSION,
        &benchmark_id,
        input_config,
    );
    export.parquet =
        crate::config::model::export::ParquetExport::build(&sm_formats, server_enabled);
    cfg.export = Some(export);

    let resolved = Resolved {
        scenario_outcome,
        ..Resolved::default()
    };

    Ok(BenchmarkRun {
        benchmark_id,
        artifact_dir: inputs.artifact_dir,
        cfg,
        cli_command: None,
        label: String::new(),
        random_seed: inputs.random_seed,
        sweep_id: None,
        trial: 0,
        variation: None,
        resolved,
        variables: serde_json::Map::new(),
    })
}

/// Resolve the legacy-AgentX submission scenario locks (`--scenario`) against the
/// resolved run config, returning the serialized [`ScenarioOutcome`] for the
/// run's `resolved` projection.
///
/// A hard [`ScenarioLockError`] (non-overridable conflict, or violations without
/// `--unsafe-override`) fails the resolution.
fn resolve_scenario_outcome(inputs: &Inputs) -> anyhow::Result<Option<serde_json::Value>> {
    use crate::agentx::scenario::{RunLockInputs, apply_scenario_locks, get_scenario};

    let Some(name) = inputs.scenario.as_deref() else {
        return Ok(None);
    };
    let spec = get_scenario(name).ok_or_else(|| anyhow::anyhow!("unknown --scenario {name:?}"))?;

    // Project the resolved run config onto the fields the invariants read. The
    // CLI does not track per-flag "explicitly set" state, so an unset field is
    // reported as non-explicit — the resolver then auto-applies the scenario
    // default rather than raising a (spurious) violation.
    let ignore_eos = inputs
        .extra
        .get("ignore_eos")
        .and_then(serde_json::Value::as_bool);
    // The detected loader is the dataset *format* (`weka_trace`, ...), which is
    // what `require_loader` keys off — not the catalog entry name. A public
    // dataset resolves through the catalog to its format; a file/inline dataset
    // uses its explicit `custom_dataset_type`. A synthetic default has neither
    // (flagged so `require_loader` can reject it).
    let loader = if inputs.public_dataset.as_deref() == Some("weka_hf") {
        Some("weka_trace".to_string())
    } else if let Some(name) = inputs.public_dataset.as_deref() {
        crate::config::model::public_catalog::lookup(name).map(|meta| meta.format.clone())
    } else {
        inputs.custom_dataset_type.clone()
    };
    let synthetic_default_dataset =
        loader.is_none() && inputs.public_dataset.is_none() && inputs.input_file.is_none();

    let recorded_agent = if spec.recorded_agent.is_some() {
        let fixture = crate::graph::recorded::agent_recording::CanonicalReplayFixture::load()
            .map_err(|error| {
                anyhow::anyhow!("loading canonical recorded-agent fixture: {error}")
            })?;
        let mut recorded =
            crate::agentx::scenario::RecordedAgentScenarioInputs::canonical(&fixture);
        let graph = inputs.recorded_agent_graph.as_ref();
        recorded.dataset_format = inputs.custom_dataset_type.clone().unwrap_or_default();
        recorded.execute_tools = graph.is_some_and(|graph| graph.execute_tools);
        recorded.virtual_clock = inputs.transport.is_dry_run();
        recorded.workers = inputs.runtime_workers.unwrap_or(1);
        recorded.cells = inputs.runtime_cells;
        recorded.allow_wrap = inputs.allow_dataset_wrap.unwrap_or(false);
        recorded.shuffle = inputs.sampling != "sequential";
        recorded.active_traces = inputs.concurrency.unwrap_or(1);
        recorded.streaming = inputs.streaming;
        recorded.use_server_token_count = inputs.use_server_token_count;
        recorded.input_truncation = inputs.max_context_length.is_some();
        recorded.sketch_metrics = inputs.sketch_metrics || env_sketch_enabled();
        recorded.pinch_image = graph
            .and_then(|graph| graph.pinch_image.clone())
            .unwrap_or_default();
        recorded.warmup = graph.is_some_and(|graph| graph.emit_warmup);
        recorded.hardware_description = inputs.hardware_description.clone();
        recorded.resume = graph.is_some_and(|graph| graph.resume);
        recorded.unsafe_override = inputs.unsafe_override;
        Some(recorded)
    } else {
        None
    };

    let lock_inputs = RunLockInputs {
        streaming: inputs.streaming,
        streaming_explicit: inputs.streaming,
        ignore_eos,
        ignore_trace_delays: inputs.ignore_trace_delays,
        ignore_trace_delays_explicit: inputs.ignore_trace_delays_explicit,
        loader,
        cache_bust: inputs.cache_bust.as_deref().and_then(|raw| match raw {
            "system_prefix" => Some(crate::agentx::cache_bust::CacheBustTarget::SystemPrefix),
            "system_suffix" => Some(crate::agentx::cache_bust::CacheBustTarget::SystemSuffix),
            "first_turn_prefix" => {
                Some(crate::agentx::cache_bust::CacheBustTarget::FirstTurnPrefix)
            }
            "first_turn_suffix" => {
                Some(crate::agentx::cache_bust::CacheBustTarget::FirstTurnSuffix)
            }
            "warmup_isolation_system" => {
                Some(crate::agentx::cache_bust::CacheBustTarget::WarmupIsolationSystem)
            }
            "warmup_isolation_first_turn" => {
                Some(crate::agentx::cache_bust::CacheBustTarget::WarmupIsolationFirstTurn)
            }
            "none" => Some(crate::agentx::cache_bust::CacheBustTarget::None),
            _ => None,
        }),
        cache_bust_explicit: inputs.cache_bust.is_some(),
        trace_idle_gap_cap_seconds: inputs.trace_idle_gap_cap_seconds,
        inter_turn_delay_cap_seconds: inputs.inter_turn_delay_cap_seconds,
        unsafe_override: inputs.unsafe_override,
        synthetic_default_dataset,
        recorded_agent,
    };

    let outcome = apply_scenario_locks(&spec, &lock_inputs)?;
    Ok(Some(serde_json::to_value(&outcome)?))
}

/// Resolve the effective WEKA reconstruction semantics for the run: an explicit
/// `--weka-semantics` flag always wins; otherwise an agentic-replay scenario
/// selects `legacy` (the byte-exact AgentX path), and everything else defers to
/// the graph-ir default (`None`). Authored onto the config so the engine's graph
/// workload factory can branch.
fn resolve_weka_semantics(inputs: &Inputs) -> Option<String> {
    if let Some(flag) = inputs.weka_semantics.as_deref() {
        return Some(flag.to_string());
    }
    if let Some(name) = inputs.scenario.as_deref()
        && let Some(spec) = crate::agentx::scenario::get_scenario(name)
        && spec.timing_mode == "agentic_replay"
    {
        return Some("legacy".to_string());
    }
    None
}

/// Resolve the effective single-coordinator worker-assignment policy.
///
/// An explicit `--hop-routing`/`runtime.hop_routing` always wins. Absent, a
/// [`ConnectionReuse::StickyUserSessions`] run defaults to [`HopRouting::Sticky`]
/// (one worker per session keeps the sticky connection pool warm); every other
/// reuse strategy defaults to [`HopRouting::RoundRobin`]. The value is inert
/// unless the run is `global-hop` or `global-push` with `workers > 1`.
pub(crate) fn resolve_hop_routing(
    explicit: Option<HopRouting>,
    connection_reuse: ConnectionReuse,
) -> HopRouting {
    explicit.unwrap_or({
        if connection_reuse == ConnectionReuse::StickyUserSessions {
            HopRouting::Sticky
        } else {
            HopRouting::RoundRobin
        }
    })
}

/// Materialize a `--scenario`'s submission locks onto `inputs` so BOTH weka
/// reconstruction runtimes honor them.
///
/// The legacy AgentX path hardcodes `ignore_eos`, the `t*` window, and the
/// cache-bust target in `lower_legacy_agentic`, so it never consulted `inputs`.
/// The graph-ir path instead composes its wire body from `inputs.extra`
/// (`ignore_eos` reaches the payload only through `endpoint.extra` -> `merge_extra`)
/// and derives its phase list + `t*` snapshot from `inputs` (an unbound
/// AgentX lane-prime `"warmup"` phase plus the recorded-graph synthesis block).
/// Without this step a `--scenario inferencex-agentx-mvp --weka-semantics graph-ir`
/// run drops `ignore_eos` from the body and runs a single unprimed profiling
/// phase (no `t*` lane-prime warmup barrier). Idempotent and conservative: only
/// fills values the user left unset, so `resolve_scenario_outcome`'s
/// explicit-conflict checks still fire.
fn apply_scenario_graph_locks(
    inputs: &mut Inputs,
    weka_semantics: Option<&str>,
) -> anyhow::Result<()> {
    let Some(name) = inputs.scenario.clone() else {
        return Ok(());
    };
    let Some(spec) = crate::agentx::scenario::get_scenario(&name) else {
        return Ok(());
    };

    // GAP 1 — scenario feature flags into the composed request body. `ignore_eos`
    // reaches the endpoint wire payload only via `inputs.extra`; insert only when
    // absent so an explicit `ignore_eos:false` still surfaces as a lock conflict.
    if spec.require_ignore_eos && !inputs.extra.contains_key("ignore_eos") {
        inputs
            .extra
            .insert("ignore_eos".to_string(), serde_json::Value::Bool(true));
    }

    // GAP 2 — both Weka arms consume the global idle guard, while only graph-ir
    // needs the `t*` snapshot window and warmup barrier that legacy synthesizes
    // inside its lowering.
    let is_graph_ir = !matches!(weka_semantics, Some("legacy") | Some("agentx"));
    let min = spec.default_trajectory_start_min_ratio.unwrap_or(0.0);
    let max = spec.default_trajectory_start_max_ratio.unwrap_or(1.0);
    apply_scenario_synthesis(inputs, &spec, min, max, is_graph_ir)?;
    if is_graph_ir {
        inputs.trajectory_start_min_ratio = min;
        inputs.trajectory_start_max_ratio = max;
        // Author an AgentX lane-prime warmup barrier (excluded from results).
        // Unbound (no request/session/duration) under the scenario t* window
        // warms only `concurrency` in-flight lanes at the turn before t*, then
        // hands those lanes into profiling — matching legacy AgentX warmup, not
        // a full-corpus one-pass. Only synthesize when the user has not authored
        // their own warmup.
        if inputs.warmup.is_none() {
            inputs.warmup = Some(Warmup {
                concurrency: inputs.concurrency,
                rate: None,
                requests: None,
                sessions: None,
                prefill_concurrency: None,
                rate_mode: None,
                concurrency_ramp: None,
                rate_ramp: None,
                prefill_ramp: None,
                duration: None,
                grace_period: None,
            });
        }
    }
    Ok(())
}

/// Apply scenario-wide replay defaults, then overlay Graph-IR reconstruction
/// synthesis onto the `synthesis` block the graph-input adapter reads
/// (`TraceSynthesisSpec`). Preserves any user-authored `--synthesis-*` values and
/// supplies the required (non-defaulted) spec fields at identity when absent.
fn apply_scenario_synthesis(
    inputs: &mut Inputs,
    spec: &crate::agentx::scenario::ScenarioSpec,
    min: f64,
    max: f64,
    is_graph_ir: bool,
) -> anyhow::Result<()> {
    if inputs.system_idle_gap_cap_seconds.is_none() {
        inputs.system_idle_gap_cap_seconds = spec.system_idle_gap_cap_seconds;
    }
    if !is_graph_ir {
        return Ok(());
    }

    use crate::agentx::cache_bust::CacheBustTarget;
    let num = |v: f64| -> anyhow::Result<serde_json::Value> {
        serde_json::Number::from_f64(v)
            .map(serde_json::Value::Number)
            .ok_or_else(|| anyhow::anyhow!("scenario synthesis value must be finite, got {v}"))
    };
    // A scenario that locks none of the synthesis-bearing knobs contributes
    // nothing here, and materializing the block anyway is not inert: the direct
    // `agent_recording` graph input (`RecordedAgentDatasetInput`,
    // `engine/graph_input.rs`) is `deny_unknown_fields` and has no `synthesis`
    // field, so an injected block fails decode outright. That makes
    // `--scenario recorded-agent-default` (whose four synthesis fields are all
    // `None`) unrunnable on the graph-ir arm. Only overlay when the scenario
    // actually locks something, or the user authored `--synthesis-*` themselves.
    let scenario_locks_synthesis = spec.default_trajectory_start_min_ratio.is_some()
        || spec.default_trajectory_start_max_ratio.is_some()
        || spec.trace_idle_gap_cap_seconds.is_some()
        || spec.require_cache_bust.is_some();
    if !scenario_locks_synthesis && inputs.synthesis.is_none() {
        return Ok(());
    }
    let mut m = match inputs.synthesis.take() {
        Some(serde_json::Value::Object(existing)) => existing,
        _ => {
            // Required (non-`serde(default)`) `TraceSynthesisSpec` fields, at
            // identity so no reconstruction transform is applied.
            let mut base = serde_json::Map::new();
            base.insert("speedup_ratio".to_string(), num(1.0)?);
            base.insert("prefix_len_multiplier".to_string(), num(1.0)?);
            base.insert(
                "prefix_root_multiplier".to_string(),
                serde_json::Value::from(1u64),
            );
            base.insert("prompt_len_multiplier".to_string(), num(1.0)?);
            base.insert("output_len_multiplier".to_string(), num(1.0)?);
            base.insert(
                "dataset_sampling_strategy".to_string(),
                serde_json::Value::String("sequential".to_string()),
            );
            base
        }
    };
    m.insert("trajectory_start_min_ratio".to_string(), num(min)?);
    m.insert("trajectory_start_max_ratio".to_string(), num(max)?);
    m.insert(
        "t_star_random_seed".to_string(),
        serde_json::Value::from(inputs.random_seed.unwrap_or(0)),
    );
    if let Some(idle) = spec.trace_idle_gap_cap_seconds {
        m.insert("idle_gap_cap_seconds".to_string(), num(idle)?);
    }
    // Only `first_turn_prefix` round-trips through `CacheBustTarget::parse`; any
    // other target stays absent (disabled) rather than silently misprojecting.
    if let Some(CacheBustTarget::FirstTurnPrefix) = spec.require_cache_bust {
        m.insert(
            "cache_bust_target".to_string(),
            serde_json::Value::String("first_turn_prefix".to_string()),
        );
    }
    inputs.synthesis = Some(serde_json::Value::Object(m));
    Ok(())
}

/// Build one phase from resolved axes. A request rate selects a Poisson arrival
/// phase (with optional concurrency cap); otherwise a fixed-concurrency phase.
#[allow(clippy::too_many_arguments)]
fn build_phase(
    name: &str,
    exclude_from_results: bool,
    default_concurrency: u32,
    rate: Option<f64>,
    rate_mode: Option<&str>,
    smoothness: Option<f64>,
    concurrency: Option<u32>,
    requests: Option<u64>,
    sessions: Option<u64>,
    duration: Option<f64>,
    grace_period: Option<f64>,
) -> Phase {
    let kind = if let Some(rate) = rate {
        match rate_mode {
            Some("gamma") => PhaseKind::Gamma {
                rate,
                concurrency,
                smoothness,
            },
            Some("constant") => PhaseKind::Constant { rate, concurrency },
            // Poisson is the default arrival distribution.
            _ => PhaseKind::Poisson { rate, concurrency },
        }
    } else {
        PhaseKind::Concurrency {
            concurrency: concurrency.unwrap_or(default_concurrency),
        }
    };
    let role = if exclude_from_results {
        PhaseRole::Warmup
    } else {
        PhaseRole::Profiling
    };
    Phase {
        common: PhaseCommon {
            timing_mode: None,
            name: name.to_string(),
            kind: Some(role),
            exclude_from_results,
            seamless: false,
            requests,
            sessions,
            duration,
            prefill_concurrency: None,
            grace_period,
            concurrency_ramp: None,
            prefill_ramp: None,
            rate_ramp: None,
            cancellation: None,
            agentic_cache_warmup_duration: None,
            agentic_warmup_grace_period: None,
            failed_request_threshold: None,
            adaptive_scale: None,
            rate_series: None,
        },
        kind,
    }
}

/// A linear ramp of the given duration (the default ramp strategy).
fn linear_ramp(duration: f64) -> crate::config::model::phase::Ramp {
    crate::config::model::phase::Ramp {
        duration,
        strategy: "linear".to_string(),
    }
}

/// Make a dataset path absolute without resolving symlinks.
fn absolute_path(path: &std::path::Path) -> String {
    if path.is_absolute() {
        return path.to_string_lossy().into_owned();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(path).to_string_lossy().into_owned(),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

/// Add `http://` when a base URL omits its scheme.
pub(crate) fn normalize_url(url: &str) -> String {
    if url.contains("://") {
        url.to_string()
    } else {
        format!("http://{url}")
    }
}

/// Return whether `AIPERF_METRICS_SKETCH` enables sketch retention.
fn env_sketch_enabled() -> bool {
    std::env::var("AIPERF_METRICS_SKETCH")
        .map(|v| is_truthy_env(&v))
        .unwrap_or(false)
}

/// Parse `1`/`true`/`t`/`yes`/`y`/`on`, ignoring case and whitespace.
pub(crate) fn is_truthy_env(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "t" | "yes" | "y" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::{is_fake_model_name, is_truthy_env, resolve_hop_routing};
    use crate::config::model::HopRouting;
    use crate::config::model::endpoint::ConnectionReuse;

    #[test]
    fn hop_routing_defaults_to_sticky_under_sticky_user_sessions() {
        assert_eq!(
            resolve_hop_routing(None, ConnectionReuse::StickyUserSessions),
            HopRouting::Sticky
        );
    }

    #[test]
    fn hop_routing_explicit_wins_over_sticky_default() {
        assert_eq!(
            resolve_hop_routing(
                Some(HopRouting::RoundRobin),
                ConnectionReuse::StickyUserSessions
            ),
            HopRouting::RoundRobin
        );
    }

    #[test]
    fn hop_routing_defaults_to_round_robin_under_pooled() {
        assert_eq!(
            resolve_hop_routing(None, ConnectionReuse::Pooled),
            HopRouting::RoundRobin
        );
    }

    #[test]
    fn hop_routing_explicit_least_loaded_under_pooled() {
        assert_eq!(
            resolve_hop_routing(Some(HopRouting::LeastLoaded), ConnectionReuse::Pooled),
            HopRouting::LeastLoaded
        );
    }

    #[test]
    fn fake_model_name_is_classified() {
        for t in [
            "mock-model",
            "test-model",
            "mock-llama",
            "Test_Model_v2",
            "fake",
            "MOCK",
            "sample",
            "my-model",
            "model-id",
            "llama-fake",
        ] {
            assert!(is_fake_model_name(t), "{t:?} should be a placeholder");
        }
        for f in [
            "Qwen/Qwen3-0.6B",
            "./local-model",
            "~/model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "contestant",
            "gpt2",
            "",
        ] {
            assert!(!is_fake_model_name(f), "{f:?} should not be a placeholder");
        }
    }

    #[test]
    fn truthy_env_is_classified() {
        for t in ["1", "true", "TRUE", "t", "yes", "Y", "on", "  On  "] {
            assert!(is_truthy_env(t), "{t:?} should be truthy");
        }
        for f in ["0", "false", "no", "off", "", "2", "enabled"] {
            assert!(!is_truthy_env(f), "{f:?} should be falsy");
        }
    }
}
