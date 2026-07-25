// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Parse `profile` flags or YAML config into a [`BenchmarkRun`].
//!
//! Flag and YAML inputs normalize to `Inputs` and share `build`, keeping wire
//! defaults in one place.

use std::path::PathBuf;

use aiperf_runtime::engine::protocol::DispatchMode;

use crate::flags::ProfileFlags;
use crate::model::artifacts::Artifacts;
use crate::model::dataset::{
    AudioSpec, Dataset, Distribution, ImageSpec, PrefixPrompts, PromptSelection, Prompts, Sampling,
    Synthetic, VideoAudio, VideoSpec,
};
use crate::model::endpoint::{
    ConnectionReuse, Endpoint, EndpointType, RequestContentType, ResetKvCacheConfig,
    ServerProfilerConfig, WaitForModelMode,
};
use crate::model::metrics::Metrics;
use crate::model::models::{ModelItem, ModelStrategy, Models};
use crate::model::phase::{AdaptiveScale, Phase, PhaseCommon, PhaseKind, PhaseRole, SlaFilter};
use crate::model::rate_series::RateSeries;
use crate::model::runtime::Runtime;
use crate::model::tokenizer::Tokenizer;
use crate::model::transport::Transport;
use crate::model::{BenchmarkConfig, BenchmarkRun, Resolved};
use crate::phase_validate::{apply_cli_loadgen_overlays, normalize_and_validate_phases};

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
const DEFAULT_ISL_MEAN: f64 = 550.0;
/// Default synthetic conversation count when no request bound is given.
pub(crate) const DEFAULT_ENTRIES: u32 = 100;
/// Default request bound when no count/duration/schedule bounds the run.
const DEFAULT_REQUEST_COUNT: u64 = 10;

fn authored_prompt_selection(corpus: Option<&str>) -> Option<PromptSelection> {
    corpus.map(|corpus| PromptSelection {
        corpus: Some(corpus.to_string()),
    })
}

/// A leading warmup phase's axes.
pub(crate) struct Warmup {
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
pub(crate) struct Inputs {
    pub model_names: Vec<String>,
    pub urls: Vec<String>,
    pub endpoint_type: String,
    pub transport: crate::model::transport::Transport,
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
    pub request_content_type: Option<RequestContentType>,
    pub wait_for_model_timeout: Option<f64>,
    pub wait_for_model_mode: Option<WaitForModelMode>,
    pub wait_for_model_interval: Option<f64>,
    pub apply_chat_template: bool,
    pub prefill_concurrency: Option<u32>,
    pub prefill_ramp: Option<f64>,
    pub gpu_telemetry_enabled: bool,
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
    pub mlflow: crate::model::export::MlflowParams,
    /// W&B sink params.
    pub wandb: crate::model::export::WandbParams,
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
    /// Emit the raw request/response JSONL (`artifacts.raw`).
    pub export_raw: bool,
    /// Emit per-request HTTP trace columns (`artifacts.trace`).
    pub export_trace: bool,
    /// Emit the per-request outputs JSON (`artifacts.export_outputs_json`).
    pub export_outputs_json: bool,
    /// Mixed ISL/OSL sequence distribution (`--seq-dist`).
    pub sequence_distribution: Option<Vec<crate::model::dataset::SeqDistEntry>>,
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
    pub random_seed: Option<u64>,
    /// Per-dataset sampling seed (`dataset.random_seed`). The `--random-seed`
    /// flag sets both this and `random_seed`; a YAML top-level `randomSeed` sets
    /// only `random_seed` (the run seed), so the two are tracked separately.
    pub dataset_random_seed: Option<u64>,
    /// File-backed dataset path (mutually exclusive with the synthetic path).
    pub input_file: Option<PathBuf>,
    /// Inline file-dataset records authored directly in the config (mutually
    /// exclusive with `input_file`; emitted verbatim as `records` on the wire).
    pub inline_records: Option<serde_json::Value>,
    /// Named submission scenario (`--scenario`; `cfg.scenario`).
    pub scenario: Option<String>,
    /// WEKA reconstruction semantics (`--weka-semantics`; legacy|graph-ir).
    pub weka_semantics: Option<String>,
    /// Recorded-graph trajectory-start window lower ratio (`--trajectory-start-min-ratio`).
    pub trajectory_start_min_ratio: f64,
    /// Recorded-graph trajectory-start window upper ratio (`--trajectory-start-max-ratio`).
    pub trajectory_start_max_ratio: f64,
    /// Relax cross-field validation (`--unsafe-override`; `cfg.unsafe_override`).
    pub unsafe_override: bool,
    /// Agentic cache-warmup duration, seconds (auto-creates a warmup phase).
    pub agentic_cache_warmup_duration: Option<f64>,
    /// Rankings/rerank query-passage generation (present when a rankings flag is set).
    pub rankings: Option<crate::model::dataset::Rankings>,
    /// Accuracy-benchmark policy (present when `--accuracy-benchmark` is set).
    pub accuracy: Option<crate::model::config::Accuracy>,
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
pub(crate) struct DatasetAnalysisInputs {
    /// KV-cache block size (tokens) for the cache-reuse analysis.
    pub block_size: u32,
    /// Explicit realized-LRU capacity (blocks) sweep point, if requested.
    pub cache_blocks: Option<u64>,
    /// Emit per-conversation breakdowns.
    pub per_conversation: bool,
}

/// The default synthetic ISL distribution (`{mean, stddev}`), used when no
/// explicit distribution is authored.
pub(crate) fn default_isl() -> Distribution {
    Distribution {
        mean: Some(DEFAULT_ISL_MEAN),
        stddev: Some(0.0),
        ..Default::default()
    }
}

/// Map `--export-level` (`summary`/`records`/`raw`, or unset) to the per-record
/// format list plus the raw-JSONL flag.
pub(crate) fn export_level_formats(level: Option<&str>) -> anyhow::Result<(Vec<String>, bool)> {
    Ok(match level {
        None => (vec!["jsonl".to_string()], false),
        Some("summary") => (Vec::new(), false),
        Some("records") => (vec!["jsonl".to_string()], false),
        Some("raw") => (vec!["jsonl".to_string()], true),
        Some(other) => anyhow::bail!("unknown --export-level {other:?} (summary/records/raw)"),
    })
}

pub(crate) fn reset_kv_cache_from_flags(flags: &ProfileFlags) -> Option<ResetKvCacheConfig> {
    (flags.reset_kv_cache
        || flags.reset_kv_cache_timeout_seconds.is_some()
        || flags.reset_kv_cache_path.is_some())
    .then(|| ResetKvCacheConfig {
        timeout_seconds: flags.reset_kv_cache_timeout_seconds,
        path: flags.reset_kv_cache_path.clone(),
    })
}

pub(crate) fn server_profiler_from_flags(flags: &ProfileFlags) -> Option<ServerProfilerConfig> {
    (flags.server_profiler
        || flags.server_profiler_timeout_seconds.is_some()
        || flags.server_profiler_start_path.is_some()
        || flags.server_profiler_stop_path.is_some())
    .then(|| ServerProfilerConfig {
        timeout_seconds: flags.server_profiler_timeout_seconds,
        start_path: flags.server_profiler_start_path.clone(),
        stop_path: flags.server_profiler_stop_path.clone(),
    })
}

pub(crate) fn overlay_reset_kv_cache_config(
    target: &mut Option<ResetKvCacheConfig>,
    overlay: Option<ResetKvCacheConfig>,
) {
    let Some(overlay) = overlay else {
        return;
    };
    match target {
        Some(existing) => {
            if overlay.timeout_seconds.is_some() {
                existing.timeout_seconds = overlay.timeout_seconds;
            }
            if overlay.path.is_some() {
                existing.path = overlay.path;
            }
        }
        None => *target = Some(overlay),
    }
}

pub(crate) fn overlay_server_profiler_config(
    target: &mut Option<ServerProfilerConfig>,
    overlay: Option<ServerProfilerConfig>,
) {
    let Some(overlay) = overlay else {
        return;
    };
    match target {
        Some(existing) => {
            if overlay.timeout_seconds.is_some() {
                existing.timeout_seconds = overlay.timeout_seconds;
            }
            if overlay.start_path.is_some() {
                existing.start_path = overlay.start_path;
            }
            if overlay.stop_path.is_some() {
                existing.stop_path = overlay.stop_path;
            }
        }
        None => *target = Some(overlay),
    }
}

pub fn resolve(flags: &ProfileFlags) -> anyhow::Result<BenchmarkRun> {
    reject_sweep("--concurrency", flags.concurrency.as_deref())?;
    reject_sweep("--request-count", flags.request_count.as_deref())?;
    reject_sweep("--request-rate", flags.request_rate.as_deref())?;
    reject_sweep("--benchmark-duration", flags.benchmark_duration.as_deref())?;
    reject_sweep("--num-conversations", flags.num_conversations.as_deref())?;

    anyhow::ensure!(
        !flags.model_names.is_empty(),
        "at least one --model is required"
    );
    // A dry run opens no sockets, so a real endpoint URL is not required — default
    // a sentinel so the endpoint/profile lowering (which still wants some URL) is
    // satisfied. The fake transport never dials it.
    let urls = if flags.urls.is_empty() && flags.dry_run {
        vec!["http://dry-run.invalid".to_string()]
    } else {
        flags.urls.clone()
    };
    anyhow::ensure!(
        !urls.is_empty(),
        "at least one --url is required (omit it only with --dry-run)"
    );
    let endpoint_type = flags
        .endpoint_type
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--endpoint-type is required"))?;

    let concurrency = parse_single::<u32>("--concurrency", flags.concurrency.as_deref())?;
    let request_count = parse_single::<u64>("--request-count", flags.request_count.as_deref())?;
    let request_rate = parse_single::<f64>("--request-rate", flags.request_rate.as_deref())?;
    let user_centric_cli = match (flags.user_centric_rate, flags.num_users) {
        (Some(rate), Some(users)) => Some((rate, users)),
        _ => None,
    };
    let request_rate_series = resolve_request_rate_series(
        flags.request_rate_series.as_ref(),
        request_rate,
        user_centric_cli.is_some(),
    )?;
    let benchmark_duration =
        parse_single::<f64>("--benchmark-duration", flags.benchmark_duration.as_deref())?;
    let num_conversations =
        parse_single::<u32>("--num-conversations", flags.num_conversations.as_deref())?;
    // Explicit entries determine dataset size but do not create a session bound.
    let num_dataset_entries = parse_single::<u32>(
        "--num-dataset-entries",
        flags.num_dataset_entries.as_deref(),
    )?;

    let (records_formats, export_raw) = export_level_formats(flags.export_level.as_deref())?;
    // Fixed-schedule replays each timestamped entry once, so the request bound is
    // the schedule length (the input file's non-empty line count).
    let (fixed_schedule, request_count) = if flags.fixed_schedule {
        let path = flags
            .input_file
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--fixed-schedule requires --input-file"))?;
        let count = count_schedule_entries(path)?;
        let default_auto = flags.fixed_schedule_start_offset.is_none()
            && flags.fixed_schedule_end_offset.is_none();
        (
            Some(flags.fixed_schedule_auto_offset.unwrap_or(default_auto)),
            Some(count),
        )
    } else {
        (None, request_count)
    };
    reject_sweep("--isl", flags.isl.as_deref())?;
    reject_sweep("--osl", flags.osl.as_deref())?;
    let isl_mean = parse_single::<f64>("--isl", flags.isl.as_deref())?;
    let osl_mean = parse_single::<f64>("--osl", flags.osl.as_deref())?;

    reject_sweep("--num-sessions", flags.num_sessions.as_deref())?;
    let num_sessions = parse_single::<u32>("--num-sessions", flags.num_sessions.as_deref())?;
    let turn_delay_ms = flags.session_turn_delay_mean.map(|mean| Distribution {
        mean: Some(mean),
        stddev: Some(flags.session_turn_delay_stddev.unwrap_or(0.0)),
        ..Default::default()
    });
    let turns = flags.session_turns_mean.map(|mean| Distribution {
        mean: Some(mean),
        stddev: Some(flags.session_turns_stddev.unwrap_or(0.0)),
        ..Default::default()
    });

    let warmup = if flags.warmup_request_count.is_none()
        && flags.warmup_concurrency.is_none()
        && flags.warmup_request_rate.is_none()
        && flags.num_warmup_sessions.is_none()
        && flags.warmup_prefill_concurrency.is_none()
        && flags.warmup_concurrency_ramp_duration.is_none()
        && flags.warmup_request_rate_ramp_duration.is_none()
        && flags.warmup_duration.is_none()
        && flags.warmup_grace_period.is_none()
    {
        None
    } else {
        Some(Warmup {
            concurrency: flags.warmup_concurrency,
            rate: flags.warmup_request_rate,
            requests: flags.warmup_request_count,
            sessions: flags.num_warmup_sessions,
            prefill_concurrency: flags.warmup_prefill_concurrency,
            rate_mode: flags.warmup_arrival_pattern.clone(),
            concurrency_ramp: flags.warmup_concurrency_ramp_duration,
            rate_ramp: flags.warmup_request_rate_ramp_duration,
            prefill_ramp: flags.warmup_prefill_concurrency_ramp_duration,
            duration: flags.warmup_duration,
            grace_period: flags.warmup_grace_period,
        })
    };

    let inputs = Inputs {
        model_names: flags.model_names.clone(),
        urls,
        endpoint_type,
        transport: if flags.dry_run {
            Transport::DryRun(crate::model::transport::DryRunConfig {
                ttft_ms: flags.dry_run_ttft_ms,
                itl_ms: flags.dry_run_itl_ms,
                ttft_per_isl_token_ms: flags.dry_run_ttft_per_isl_ms,
                ttft_concurrency_quad_ms: flags.dry_run_ttft_concurrency_quad_ms,
                itl_per_osl_token_ms: flags.dry_run_itl_per_osl_ms,
                itl_concurrency_lin_ms: flags.dry_run_itl_concurrency_lin_ms,
                ttft_jitter_cv: flags.dry_run_ttft_jitter_cv,
                itl_jitter_cv: flags.dry_run_itl_jitter_cv,
                seed: flags.dry_run_seed,
                latency_model: flags.dry_run_latency_model.clone(),
                kv_utilization: flags.dry_run_kv_utilization,
                clock: flags.dry_run_clock.clone(),
            })
        } else {
            Transport::Http
        },
        streaming: flags.streaming,
        timeout_seconds: flags.request_timeout_seconds,
        use_legacy_max_tokens: flags.use_legacy_max_tokens,
        use_server_token_count: flags.use_server_token_count,
        download_video_content: flags.download_video_content,
        extra: parse_extra_inputs(&flags.extra_inputs)?,
        // Flag URLs normalize here; YAML values remain authored until sidecar construction.
        server_metrics_urls: flags
            .server_metrics
            .iter()
            .map(|u| crate::model::telemetry::normalize_metrics_url(u))
            .collect(),
        connection_reuse: flags
            .connection_reuse_strategy
            .as_deref()
            .map(parse_connection_reuse)
            .transpose()?,
        request_content_type: flags
            .request_content_type
            .as_deref()
            .map(parse_content_type)
            .transpose()?,
        wait_for_model_timeout: flags.wait_for_model_timeout,
        wait_for_model_mode: flags
            .wait_for_model_mode
            .as_deref()
            .map(parse_wait_mode)
            .transpose()?,
        wait_for_model_interval: flags.wait_for_model_interval,
        apply_chat_template: flags.apply_chat_template,
        prefill_concurrency: flags.prefill_concurrency,
        prefill_ramp: flags.prefill_concurrency_ramp_duration,
        gpu_telemetry_enabled: !flags.no_gpu_telemetry,
        // A `.csv` value selects custom metrics; all other values are scrape URLs.
        gpu_telemetry_urls: flags
            .gpu_telemetry
            .iter()
            .filter(|item| !item.to_ascii_lowercase().ends_with(".csv"))
            .cloned()
            .collect(),
        gpu_telemetry_metrics_file: flags
            .gpu_telemetry
            .iter()
            .rev()
            .find(|item| item.to_ascii_lowercase().ends_with(".csv"))
            .cloned(),
        server_metrics_enabled: !flags.no_server_metrics,
        server_metrics_formats: (!flags.server_metrics_formats.is_empty())
            .then(|| flags.server_metrics_formats.clone()),
        slos: parse_goodput(flags.goodput.as_deref())?,
        network_latency_mean: flags.network_latency_mean,
        network_latency_probe: flags
            .network_latency_automatic
            .then(|| flags.network_latency_ping_interval.unwrap_or(1.0)),
        otel_url: flags.otel_url.clone(),
        otel_provider: flags.gen_ai_provider.clone(),
        otel_resource_attributes: parse_kv(&flags.otel_resource_attributes, '=')?,
        mlflow: crate::model::export::MlflowParams {
            tracking_uri: flags.mlflow_tracking_uri.clone(),
            experiment: flags.mlflow_experiment.clone(),
            run_name: flags.mlflow_run_name.clone(),
            parent_run_id: flags.mlflow_parent_run_id.clone(),
            tags: parse_kv(&flags.mlflow_tag, ':')?,
            artifact_globs: flags.mlflow_artifact_glob.clone(),
            // MLflow logs the run's request bound as `total_expected_requests`.
            total_expected_requests: request_count.map(|n| n as f64),
        },
        wandb: crate::model::export::WandbParams {
            project: flags.wandb_project.clone(),
            entity: flags.wandb_entity.clone(),
            run_name: flags.wandb_run_name.clone(),
            tags: flags.wandb_tag.clone(),
        },
        api_key: flags.api_key.clone(),
        headers: parse_headers(&flags.headers)?,
        tokenizer_name: flags.tokenizer.clone(),
        tokenizer_revision: flags.tokenizer_revision.clone(),
        tokenizer_trust: flags.tokenizer_trust_remote_code,
        server_tokenizer_url: flags.server_tokenizer_url.clone(),
        isl: match isl_mean {
            Some(mean) => Distribution {
                mean: Some(mean),
                stddev: Some(flags.isl_stddev.unwrap_or(0.0)),
                ..Default::default()
            },
            None => default_isl(),
        },
        osl: osl_mean.map(|mean| Distribution {
            mean: Some(mean),
            stddev: Some(flags.osl_stddev.unwrap_or(0.0)),
            ..Default::default()
        }),
        turns,
        turn_delay_ratio: flags.session_delay_ratio.unwrap_or(1.0),
        turn_delay_ms,
        session_header: flags.session_header.clone(),
        proxy: flags.proxy.clone(),
        proxy_from_env: flags.proxy_from_env,
        endpoint_path: flags.custom_endpoint.clone(),
        reset_kv_cache: reset_kv_cache_from_flags(flags),
        server_profiler: server_profiler_from_flags(flags),
        records_formats,
        export_raw,
        export_trace: flags.export_http_trace,
        export_outputs_json: false,
        sequence_distribution: flags.seq_dist.as_deref().map(parse_seq_dist).transpose()?,
        batch_size: flags.batch_size.unwrap_or(1),
        sampling: flags
            .dataset_sampling_strategy
            .clone()
            .unwrap_or_else(|| "sequential".to_string()),
        entries: num_dataset_entries
            .or(num_conversations)
            .or(num_sessions)
            .or(request_count.map(|n| n as u32))
            .unwrap_or(DEFAULT_ENTRIES),
        dataset_entries: num_dataset_entries
            .or(num_conversations)
            .or(num_sessions)
            .or(request_count.map(|n| n as u32)),
        sessions: num_conversations.or(num_sessions).map(u64::from),
        concurrency,
        request_rate,
        rate_mode: flags
            .request_rate_mode
            .clone()
            .or_else(|| flags.arrival_pattern.clone()),
        smoothness: flags.arrival_smoothness.or(flags.vllm_burstiness),
        concurrency_ramp: flags.concurrency_ramp_duration,
        rate_ramp: flags.request_rate_ramp_duration,
        cancellation: match (
            flags.request_cancellation_rate,
            flags.request_cancellation_delay,
        ) {
            (Some(rate), delay) => Some((rate, delay.unwrap_or(0.0))),
            _ => None,
        },
        user_centric: user_centric_cli,
        request_rate_series,
        request_count,
        benchmark_duration,
        grace_period: flags.benchmark_grace_period,
        warmup,
        random_seed: flags.random_seed,
        dataset_random_seed: flags.random_seed,
        runtime_workers: None,
        runtime_workers_min: None,
        runtime_cells: flags.cells.unwrap_or(1),
        runtime_dispatch: flags
            .dispatch
            .is_some()
            .then(|| flags.dispatch_mode())
            .transpose()?,
        input_file: flags.input_file.clone(),
        inline_records: None,
        custom_dataset_type: flags.custom_dataset_type.clone(),
        public_dataset: flags.public_dataset.clone(),
        hf_subset: flags.hf_subset.clone(),
        hf_dataset: flags.hf_dataset.clone(),
        hf_split: flags.hf_split.clone(),
        hf_revision: flags.hf_revision.clone(),
        hf_text_column: flags.hf_text_column.clone(),
        hf_output_column: flags.hf_output_column.clone(),
        hf_output_len: flags.hf_output_len,
        hf_format: flags.hf_format.clone(),
        inter_turn_delay_cap_seconds: flags.inter_turn_delay_cap_seconds,
        prefetch_media_urls: flags.prefetch_media_urls,
        uuid_and_strip: flags.uuid_and_strip,
        replay_speedup: flags.replay_speedup,
        max_idle_gap_cap_seconds: flags.max_idle_gap_cap_seconds,
        open_loop_replay: flags.open_loop_replay && !flags.no_open_loop_replay,
        open_loop_strict: flags.open_loop_strict,
        omit_kv_hints: flags.omit_kv_hints,
        force_min_tokens: flags.force_min_tokens && !flags.no_force_min_tokens,
        fixed_schedule,
        fixed_schedule_start_offset: flags.fixed_schedule_start_offset,
        fixed_schedule_end_offset: flags.fixed_schedule_end_offset,
        model_strategy: flags
            .model_selection_strategy
            .as_deref()
            .map(parse_model_strategy)
            .transpose()?,
        slice_duration: flags.slice_duration,
        isl_block_size: flags.isl_block_size,
        prefix_reuse_fraction: flags.prefix_reuse_fraction,
        prefix_reuse_ratio: flags.prefix_reuse_ratio,
        prompt_corpus: flags.prompt_corpus.clone(),
        sketch_metrics: flags.sketch_metrics,
        steady_state: flags.steady_state,
        steady_state_fraction: flags.steady_state_fraction,
        steady_state_hybrid: flags.steady_state_hybrid,
        image_spec: build_image_spec(flags),
        audio_spec: build_audio_spec(flags),
        video_spec: build_video_spec(flags),
        adaptive_scale: build_adaptive_scale(flags, concurrency)?,
        prefix_prompts: build_prefix_prompts(flags),
        scenario: flags.scenario.clone(),
        weka_semantics: flags.weka_semantics.clone(),
        trajectory_start_min_ratio: flags.trajectory_start_min_ratio.unwrap_or(0.0),
        trajectory_start_max_ratio: flags.trajectory_start_max_ratio.unwrap_or(0.0),
        unsafe_override: flags.unsafe_override,
        agentic_cache_warmup_duration: flags.agentic_cache_warmup_duration,
        rankings: build_rankings(flags),
        accuracy: build_accuracy(flags),
        synthesis: build_synthesis(flags)?,
        dataset_filters: parse_dataset_filters(flags)?,
        // Dry-run emits the dataset-analysis artifact family unless suppressed.
        dataset_analysis: (flags.dry_run && !flags.no_dataset_analysis).then(|| {
            DatasetAnalysisInputs {
                block_size: flags.kv_block_size,
                cache_blocks: flags.kv_cache_blocks,
                per_conversation: flags.dataset_analysis_per_conversation,
            }
        }),
        phases_override: None,
        artifact_dir: flags
            .artifact_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("artifacts")),
    };
    validate_baseten_extra_input_collisions(flags)?;
    build(inputs)
}

/// Reject `--extra-inputs` keys the baseten_trace loader injects per-turn.
///
/// Ports Python's `_reject_baseten_trace_extra_input_collisions`:
/// loader-injected per-turn values (`min_tokens` from the recorded output
/// length, `hash_ids`/`block_size` KV hints) overwrite endpoint-level
/// extras, so the user's value would be silently clobbered on the wire.
/// `max_tokens` is not guarded: user extras win over the loader for that key.
///
/// Scope cut: only covers the CLI-flags path (`--extra-inputs`), not
/// `endpoint.extra` authored in a YAML config -- the shared `Inputs` bag
/// `build()` receives doesn't currently carry raw extra-inputs pairs, and
/// threading that through the YAML path safely wasn't done here.
fn validate_baseten_extra_input_collisions(flags: &ProfileFlags) -> anyhow::Result<()> {
    if flags.custom_dataset_type.as_deref() != Some("baseten_trace") {
        return Ok(());
    }
    let extra = parse_extra_inputs(&flags.extra_inputs)?;
    let mut collisions: Vec<(&str, &str)> = Vec::new();
    let force_min_tokens = flags.force_min_tokens && !flags.no_force_min_tokens;
    if force_min_tokens && extra.contains_key("min_tokens") {
        collisions.push(("min_tokens", "--no-force-min-tokens"));
    }
    if !flags.omit_kv_hints {
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

/// Build one run from normalized inputs.
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
    if set_flags.is_empty() {
        return Ok(());
    }
    let msg = format!(
        "{} is only supported by the baseten_trace loader",
        set_flags.join(", ")
    );
    if inputs.public_dataset.is_some() || inputs.input_file.is_none() {
        anyhow::bail!("{msg}; provide --input-file and --custom-dataset-type baseten_trace.");
    }
    if let Some(format) = &inputs.custom_dataset_type
        && format != "baseten_trace"
    {
        anyhow::bail!("{msg}, but --custom-dataset-type is {format}.");
    }
    Ok(())
}

pub(crate) fn build(mut inputs: Inputs) -> anyhow::Result<BenchmarkRun> {
    validate_baseten_only_trace_flags(&inputs)?;
    // Resolve legacy-AgentX scenario locks (`--scenario`) while `inputs` is still
    // whole (later lowering partially moves it). A no-op without the `agentx`
    // feature. A hard scenario-lock conflict fails resolution here.
    let scenario_outcome = resolve_scenario_outcome(&inputs)?;
    // Effective weka semantics, resolved while `inputs` is still whole.
    let weka_semantics = resolve_weka_semantics(&inputs);
    // The agentic_replay (legacy weka) timing mode is a single coherent driver:
    // one workload instance owns the per-tree join gate, session-tree registry,
    // and recycle cursor. It runs global-dispatch, single-worker, non-cellular.
    // Cellular execution (`--cells > 1`) would partition a trajectory tree's root
    // and subagent children across cell processes, breaking join gating — reject
    // it with a clear error (use `--weka-semantics graph-ir` for cellular weka).
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
    // Restrict `--agentic-cache-warmup-duration` to the agentic_replay (legacy
    // weka) timing mode. The accelerated cache-warmup substage is consumed solely
    // by the agentic_replay lowering; on any other run the value is silently
    // dropped, so an unguarded flag is an invisible no-op. Reject it instead
    // (ports Python's `validate_agentic_cache_warmup`). The resolved timing mode
    // is agentic_replay exactly when `weka_semantics` is legacy/agentx (the
    // scenario-declared or `--weka-semantics`-forced legacy path).
    if inputs.agentic_cache_warmup_duration.is_some()
        && !matches!(weka_semantics.as_deref(), Some("legacy") | Some("agentx"))
    {
        anyhow::bail!(
            "--agentic-cache-warmup-duration requires the agentic_replay timing mode \
             (set by --scenario inferencex-agentx-mvp or --weka-semantics legacy); \
             the resolved timing mode is not agentic_replay."
        );
    }
    let loadgen_overlay = crate::phase_validate::LoadgenOverlay::from_inputs(&inputs);
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
    let endpoint = Endpoint {
        urls: inputs.urls.iter().map(|u| normalize_url(u)).collect(),
        endpoint_type: EndpointType(inputs.endpoint_type),
        streaming: inputs.streaming,
        use_legacy_max_tokens: inputs.use_legacy_max_tokens,
        use_server_token_count: inputs.use_server_token_count,
        timeout_seconds: inputs.timeout_seconds.unwrap_or(DEFAULT_TIMEOUT_SECONDS),
        connection_reuse: inputs.connection_reuse.unwrap_or(ConnectionReuse::Pooled),
        ssl_verify: true,
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
        Dataset::Public(crate::model::dataset::PublicDataset {
            name: id.clone(),
            format: inputs.hf_format.clone().unwrap_or_else(|| "hf".to_string()),
            source: serde_json::Value::Object(source),
            options,
            sampling: Sampling(inputs.sampling.clone()),
            entries: inputs.dataset_entries,
            random_seed: inputs.dataset_random_seed,
            prompts: authored_prompt_selection(inputs.prompt_corpus.as_deref()),
            prefetch_media_urls: inputs.prefetch_media_urls,
        })
    } else if let Some(name) = &inputs.public_dataset {
        let meta = crate::model::public_catalog::lookup(name)
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
        if let Some(max) = crate::model::public_catalog::max_conversations(
            meta,
            Some(inputs.entries),
            inputs.request_count,
        ) {
            options.insert("max_conversations".to_string(), serde_json::json!(max));
        }
        let mut source = meta.source.clone();
        if let (Some(subset), Some(obj)) = (&inputs.hf_subset, source.as_object_mut()) {
            obj.insert("subset".to_string(), serde_json::json!(subset));
        }
        Dataset::Public(crate::model::dataset::PublicDataset {
            name: name.clone(),
            format: meta.format.clone(),
            source,
            options,
            sampling: Sampling(inputs.sampling.clone()),
            entries: inputs.dataset_entries,
            random_seed: inputs.dataset_random_seed,
            prompts: authored_prompt_selection(inputs.prompt_corpus.as_deref()),
            prefetch_media_urls: inputs.prefetch_media_urls,
        })
    } else if inputs.input_file.is_some() || inputs.inline_records.is_some() {
        if inputs.uuid_and_strip && endpoint_type_for_dataset_validation != "chat" {
            anyhow::bail!("--uuid-and-strip requires endpoint type 'chat'");
        }
        Dataset::File(crate::model::dataset::FileDataset {
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
                match format {
                    "mooncake_trace" => {
                        o.insert("block_size".to_string(), serde_json::json!(512));
                    }
                    "bailian_trace" => {
                        o.insert("block_size".to_string(), serde_json::json!(16));
                    }
                    _ => {}
                }
                if let Some(cap) = inputs.inter_turn_delay_cap_seconds {
                    o.insert(
                        "inter_turn_delay_cap_seconds".to_string(),
                        serde_json::json!(cap),
                    );
                }
                if inputs.uuid_and_strip {
                    o.insert("uuid_and_strip".to_string(), serde_json::json!(true));
                }
                if let Some(speedup) = inputs.replay_speedup {
                    o.insert("replay_speedup".to_string(), serde_json::json!(speedup));
                }
                if let Some(cap) = inputs.max_idle_gap_cap_seconds {
                    o.insert(
                        "max_idle_gap_cap_seconds".to_string(),
                        serde_json::json!(cap),
                    );
                }
                if !inputs.open_loop_replay {
                    o.insert("open_loop_replay".to_string(), serde_json::json!(false));
                }
                if inputs.open_loop_strict {
                    o.insert("open_loop_strict".to_string(), serde_json::json!(true));
                }
                if inputs.omit_kv_hints {
                    o.insert("omit_kv_hints".to_string(), serde_json::json!(true));
                }
                if !inputs.force_min_tokens {
                    o.insert("force_min_tokens".to_string(), serde_json::json!(false));
                }
                if let Some(block_size) = inputs.isl_block_size {
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
            synthesis: inputs.synthesis.clone(),
            prefetch_media_urls: inputs.prefetch_media_urls,
        })
    } else {
        Dataset::Synthetic(Synthetic {
            prompts: Prompts {
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
    let profiling = if let Some((rate, users)) = inputs.user_centric {
        Phase {
            common: PhaseCommon {
                name: "profiling".to_string(),
                kind: Some(PhaseRole::Profiling),
                exclude_from_results: false,
                seamless: false,
                requests: inputs.request_count,
                sessions: inputs.sessions,
                duration: inputs.benchmark_duration,
                prefill_concurrency: None,
                grace_period: inputs.grace_period,
                concurrency_ramp: None,
                prefill_ramp: None,
                rate_ramp: None,
                cancellation: None,
                agentic_cache_warmup_duration: None,
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
            inputs.grace_period,
        );
        phase.common.adaptive_scale = inputs.adaptive_scale.clone();
        phase.common.prefill_concurrency = inputs.prefill_concurrency;
        phase.common.prefill_ramp = inputs.prefill_ramp.map(linear_ramp);
        phase.common.concurrency_ramp = inputs.concurrency_ramp.map(linear_ramp);
        phase.common.rate_ramp = inputs.rate_ramp.map(linear_ramp);
        phase.common.cancellation = inputs
            .cancellation
            .map(|(rate, delay)| crate::model::phase::Cancellation { rate, delay });
        phase.common.rate_series = inputs.request_rate_series.clone();
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
    // A fixed mean takes precedence over active network probing.
    let mut network_latency_cfg = crate::model::telemetry::NetworkLatencyConfig::default();
    let network_latency_sidecar = if no_server_sidecars {
        None
    } else if let Some(mean_ms) = inputs.network_latency_mean {
        network_latency_cfg.enabled = true;
        network_latency_cfg.mean_ms = Some(mean_ms);
        Some(crate::model::telemetry::NetworkLatencySidecar::fixed(
            mean_ms,
        ))
    } else if let Some(ping) = inputs.network_latency_probe {
        network_latency_cfg.enabled = true;
        network_latency_cfg.ping_interval = ping;
        Some(crate::model::telemetry::NetworkLatencySidecar::probe(ping))
    } else {
        None
    };
    let sidecars = crate::model::telemetry::Sidecars {
        gpu_telemetry: gpu_enabled.then(|| {
            crate::model::telemetry::GpuTelemetrySidecar::default_dcgm(
                &inputs.gpu_telemetry_urls,
                inputs.gpu_telemetry_metrics_file.as_deref(),
            )
        }),
        server_metrics: server_enabled.then(|| {
            let mut all_urls = endpoint_urls.clone();
            all_urls.extend(inputs.server_metrics_urls.iter().cloned());
            let sc = crate::model::telemetry::ServerMetricsSidecar::from_endpoint_urls(&all_urls);
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
        content_server: crate::model::telemetry::ContentServerSidecar::from_env(),
    };
    let gpu_cfg = crate::model::telemetry::GpuTelemetryConfig {
        enabled: inputs.gpu_telemetry_enabled,
        urls: inputs.gpu_telemetry_urls.clone(),
        ..Default::default()
    };
    let server_cfg = crate::model::telemetry::ServerMetricsConfig {
        enabled: inputs.server_metrics_enabled,
        // Config preserves authored URLs; sidecar construction normalizes them.
        urls: inputs.server_metrics_urls.clone(),
        formats: inputs
            .server_metrics_formats
            .clone()
            .unwrap_or_else(|| crate::model::telemetry::ServerMetricsConfig::default().formats),
        ..Default::default()
    };
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
        }),
        metrics: Some(Metrics {
            slos: inputs.slos.clone(),
            slice_duration_seconds: inputs.slice_duration,
            sketch: sketch_metrics.then_some(true),
            steady_state: inputs
                .steady_state
                .then(|| crate::model::metrics::SteadyState {
                    enabled: true,
                    fraction: inputs.steady_state_fraction,
                    hybrid_latency: inputs.steady_state_hybrid,
                }),
        }),
        slos: (!inputs.slos.is_empty()).then(|| inputs.slos.clone()),
        artifacts: Some({
            // Sketch retention and DynoSim do not provide per-record values.
            let per_record = !sketch_metrics && !is_dynosim;
            let has = |f: &str| inputs.records_formats.iter().any(|x| x == f);
            Artifacts {
                trace: inputs.export_trace && per_record,
                inputs_path: "inputs.json".to_string(),
                // `raw` forces the base JSONL on even when the format list omits it.
                records_path: (per_record && (has("jsonl") || inputs.export_raw))
                    .then(|| "profile_export.jsonl".to_string()),
                records_csv_path: (per_record && has("csv"))
                    .then(|| "profile_export_records.csv".to_string()),
                records_parquet_path: (per_record && has("parquet"))
                    .then(|| "profile_export.parquet".to_string()),
                raw_path: (per_record && inputs.export_raw)
                    .then(|| "profile_export_raw.jsonl".to_string()),
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
                dataset_analysis_per_conversation: inputs
                    .dataset_analysis
                    .as_ref()
                    .is_some_and(|a| a.per_conversation),
                ..Default::default()
            }
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
        trajectory_start_max_ratio: inputs.trajectory_start_max_ratio,
        trajectory_start_min_ratio: inputs.trajectory_start_min_ratio,
        unsafe_override: inputs.unsafe_override,
    };

    let benchmark_id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
    // The genai-perf-v1 envelope requires an echoed config projection.
    let mut input_config = serde_json::to_value(&cfg).unwrap_or(serde_json::Value::Null);
    // Redaction applies only to the export copy, not runtime authentication.
    crate::redact::redact_input_config(&mut input_config);
    let mut export = crate::model::export::Export::build(
        &endpoint_type,
        true,
        &benchmark_id,
        input_config,
        serde_json::json!({}),
    );
    // The per-record OTLP sink is unavailable under sketch retention.
    if let Some(url) = &inputs.otel_url
        && !sketch_metrics
    {
        export.otel = Some(crate::model::export::OtelExport::build(
            url,
            &benchmark_id,
            &endpoint_type,
            &primary_model,
            inputs.otel_provider.as_deref(),
            &inputs.otel_resource_attributes,
        ));
    }
    export.mlflow = crate::model::export::MlflowExport::build(&inputs.mlflow, &benchmark_id);
    export.wandb = crate::model::export::WandbExport::build(&inputs.wandb, &benchmark_id);
    // Export formats use server-metrics config before export insertion.
    let sm_formats = cfg
        .server_metrics
        .as_ref()
        .map(|s| s.formats.clone())
        .unwrap_or_default();
    let sm_input_config = serde_json::to_value(&cfg).unwrap_or(serde_json::Value::Null);
    export.server_metrics = crate::model::export::ServerMetricsExport::build(
        &sm_formats,
        server_enabled,
        crate::model::export::AIPERF_V1_VERSION,
        &benchmark_id,
        sm_input_config,
    );
    export.parquet = crate::model::export::ParquetExport::build(&sm_formats, server_enabled);
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
/// Compiled only under the `agentx` feature; a hard [`ScenarioLockError`]
/// (non-overridable conflict, or violations without `--unsafe-override`) fails
/// the resolution. Without the feature this is a no-op returning `None`, so a
/// lean build silently ignores `--scenario`.
#[cfg(feature = "agentx")]
fn resolve_scenario_outcome(inputs: &Inputs) -> anyhow::Result<Option<serde_json::Value>> {
    use aiperf_runtime::agentx::scenario::{apply_scenario_locks, get_scenario, RunLockInputs};

    let Some(name) = inputs.scenario.as_deref() else {
        return Ok(None);
    };
    let spec = get_scenario(name)
        .ok_or_else(|| anyhow::anyhow!("unknown --scenario {name:?}"))?;

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
    let loader = if let Some(name) = inputs.public_dataset.as_deref() {
        crate::model::public_catalog::lookup(name).map(|meta| meta.format.clone())
    } else {
        inputs.custom_dataset_type.clone()
    };
    let synthetic_default_dataset =
        loader.is_none() && inputs.public_dataset.is_none() && inputs.input_file.is_none();

    let lock_inputs = RunLockInputs {
        streaming: inputs.streaming,
        streaming_explicit: inputs.streaming,
        ignore_eos,
        ignore_trace_delays: false,
        ignore_trace_delays_explicit: false,
        loader,
        cache_bust: None,
        cache_bust_explicit: false,
        unsafe_override: inputs.unsafe_override,
        synthetic_default_dataset,
    };

    let outcome = apply_scenario_locks(&spec, &lock_inputs)?;
    Ok(Some(serde_json::to_value(&outcome)?))
}

/// No-op scenario resolution for lean builds without the `agentx` feature.
#[cfg(not(feature = "agentx"))]
fn resolve_scenario_outcome(_inputs: &Inputs) -> anyhow::Result<Option<serde_json::Value>> {
    Ok(None)
}

/// Resolve the effective WEKA reconstruction semantics for the run: an explicit
/// `--weka-semantics` flag always wins; otherwise an agentic-replay scenario
/// selects `legacy` (the byte-exact AgentX path), and everything else defers to
/// the graph-ir default (`None`). Authored onto the config so the engine's graph
/// workload factory can branch. Only the `agentx` build can select `legacy`.
#[cfg(feature = "agentx")]
fn resolve_weka_semantics(inputs: &Inputs) -> Option<String> {
    if let Some(flag) = inputs.weka_semantics.as_deref() {
        return Some(flag.to_string());
    }
    if let Some(name) = inputs.scenario.as_deref()
        && let Some(spec) = aiperf_runtime::agentx::scenario::get_scenario(name)
        && spec.timing_mode == "agentic_replay"
    {
        return Some("legacy".to_string());
    }
    None
}

/// Without the `agentx` feature, only an explicit flag threads through (the
/// engine rejects `legacy` at selection since the legacy path is compiled out).
#[cfg(not(feature = "agentx"))]
fn resolve_weka_semantics(inputs: &Inputs) -> Option<String> {
    inputs.weka_semantics.clone()
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
            adaptive_scale: None,
            rate_series: None,
        },
        kind,
    }
}

/// Resolve `--request-rate-series` against mutually exclusive scalar rate flags.
pub(crate) fn resolve_request_rate_series(
    path: Option<&PathBuf>,
    scalar_rate: Option<f64>,
    user_centric: bool,
) -> anyhow::Result<Option<RateSeries>> {
    let Some(path) = path else {
        return Ok(None);
    };
    if scalar_rate.is_some() {
        anyhow::bail!("--request-rate and --request-rate-series are mutually exclusive");
    }
    if user_centric {
        anyhow::bail!("--request-rate-series is not supported with user-centric scheduling");
    }
    Ok(Some(RateSeries::from_json_path(path)?))
}

/// Parse repeatable `Name:value` header flags, splitting on the first colon.
fn parse_headers(raw: &[String]) -> anyhow::Result<std::collections::BTreeMap<String, String>> {
    let mut headers = std::collections::BTreeMap::new();
    for entry in raw {
        let (name, value) = entry
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("invalid --header {entry:?}; expected `Name:value`"))?;
        headers.insert(name.trim().to_string(), value.trim().to_string());
    }
    Ok(headers)
}

/// Parse repeatable `--extra-inputs key:value` into a typed JSON map.
fn parse_extra_inputs(
    raw: &[String],
) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let mut extra = serde_json::Map::new();
    for entry in raw {
        let (key, value) = entry.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("invalid --extra-inputs {entry:?}; expected key:value")
        })?;
        let v = if let Ok(i) = value.parse::<i64>() {
            serde_json::json!(i)
        } else if let Ok(f) = value.parse::<f64>() {
            serde_json::json!(f)
        } else if value == "true" || value == "false" {
            serde_json::json!(value == "true")
        } else {
            serde_json::json!(value)
        };
        extra.insert(key.to_string(), v);
    }
    Ok(extra)
}

/// Parse `--goodput` (`metric:threshold` space-separated) into an SLO map.
fn parse_goodput(
    value: Option<&str>,
) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let mut slos = serde_json::Map::new();
    let Some(value) = value else {
        return Ok(slos);
    };
    for entry in value.split_whitespace() {
        let (metric, threshold) = entry.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("invalid --goodput entry {entry:?}; expected metric:threshold")
        })?;
        let threshold: f64 = threshold
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid --goodput threshold in {entry:?}: {e}"))?;
        slos.insert(metric.to_string(), serde_json::json!(threshold));
    }
    Ok(slos)
}

/// Build the adaptive-scale controller when `--adaptive-scale` is set. Requires
/// `--adaptive-sustain-duration` and at least one `--adaptive-scale-sla`.
fn build_adaptive_scale(
    flags: &ProfileFlags,
    concurrency: Option<u32>,
) -> anyhow::Result<Option<AdaptiveScale>> {
    if !flags.adaptive_scale {
        return Ok(None);
    }
    let sustain = flags
        .adaptive_sustain_duration
        .ok_or_else(|| anyhow::anyhow!("--adaptive-scale requires --adaptive-sustain-duration"))?;
    anyhow::ensure!(
        !flags.adaptive_scale_sla.is_empty(),
        "--adaptive-scale requires at least one --adaptive-scale-sla"
    );
    let control_variable = flags
        .adaptive_control_variable
        .clone()
        .unwrap_or_else(|| "concurrency".to_string());
    // For the concurrency axis the ceiling defaults to the phase concurrency.
    let maximum = flags
        .adaptive_control_max
        .or_else(|| concurrency.map(i64::from))
        .ok_or_else(|| anyhow::anyhow!("--adaptive-scale could not resolve a maximum"))?;
    let sla_filters = flags
        .adaptive_scale_sla
        .iter()
        .map(|s| parse_sla_filter(s))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Some(AdaptiveScale {
        control_variable,
        // Flag bounds serialize as integers.
        minimum: flags.adaptive_control_min.unwrap_or(1).into(),
        maximum: maximum.into(),
        assessment_period_seconds: flags.adaptive_assessment_period.unwrap_or(30.0),
        sustain_duration_seconds: sustain,
        min_completed_requests: 1,
        strategy_type: "ramp_until_fail".to_string(),
        step_policy: "sla_margin".to_string(),
        base_step: 10,
        max_step_multiplier: 4,
        step_percent: 25.0,
        sla_filters,
    }))
}

/// Parse `metric:stat:op:threshold` into an SLA filter.
fn parse_sla_filter(s: &str) -> anyhow::Result<SlaFilter> {
    let parts: Vec<&str> = s.split(':').collect();
    anyhow::ensure!(
        parts.len() == 4,
        "invalid --adaptive-scale-sla {s:?}; expected metric:stat:op:threshold"
    );
    Ok(SlaFilter {
        metric_tag: parts[0].to_string(),
        stat: parts[1].to_string(),
        op: parts[2].to_string(),
        threshold: parts[3]
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid SLA threshold in {s:?}: {e}"))?,
    })
}

/// A default synthetic media dimension (`{value: 512}`) used when unset.
pub(crate) fn default_media_dim() -> Distribution {
    Distribution {
        value: Some(512.0),
        ..Default::default()
    }
}

/// Resolve media batch size from an explicit value or shape-triggering fields.
fn implicit_media_batch(explicit: Option<u32>, shape_trigger: bool) -> u32 {
    match explicit {
        Some(b) => b,
        None if shape_trigger => 1,
        None => 0,
    }
}

fn build_image_spec(flags: &ProfileFlags) -> Option<ImageSpec> {
    let any = flags.image_width_mean.is_some()
        || flags.image_height_mean.is_some()
        || flags.image_batch_size.is_some()
        || flags.image_format.is_some()
        || flags.image_source.is_some()
        || flags.image_source_sampling.is_some();
    if !any {
        return None;
    }
    let dim = |mean: Option<f64>, stddev: Option<f64>| match mean {
        Some(mean) => Distribution {
            mean: Some(mean),
            stddev: Some(stddev.unwrap_or(0.0)),
            ..Default::default()
        },
        None => default_media_dim(),
    };
    Some(ImageSpec {
        batch_size: implicit_media_batch(
            flags.image_batch_size,
            flags.image_width_mean.is_some()
                || flags.image_width_stddev.is_some()
                || flags.image_height_mean.is_some()
                || flags.image_height_stddev.is_some()
                || flags.image_source.is_some()
                || flags.image_source_sampling.is_some(),
        ),
        format: flags
            .image_format
            .clone()
            .unwrap_or_else(|| "jpeg".to_string()),
        height: dim(flags.image_height_mean, flags.image_height_stddev),
        width: dim(flags.image_width_mean, flags.image_width_stddev),
        source: flags
            .image_source
            .clone()
            .unwrap_or_else(|| "noise".to_string()),
        source_sampling: flags
            .image_source_sampling
            .clone()
            .unwrap_or_else(|| "random-with-replacement".to_string()),
    })
}

/// Parse `--dataset-filter key=value`, rejecting duplicates and non-public use.
fn parse_dataset_filters(
    flags: &ProfileFlags,
) -> anyhow::Result<Option<serde_json::Map<String, serde_json::Value>>> {
    let Some(entries) = flags.dataset_filter.as_ref().filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    anyhow::ensure!(
        flags.public_dataset.is_some() || flags.hf_dataset.is_some(),
        "--dataset-filter requires --public-dataset or --hf-dataset"
    );
    let mut map = serde_json::Map::new();
    for entry in entries {
        let (key, value) = entry
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--dataset-filter {entry:?} must be key=value"))?;
        anyhow::ensure!(
            !map.contains_key(key),
            "--dataset-filter key {key:?} specified more than once"
        );
        map.insert(
            key.to_string(),
            serde_json::Value::String(value.to_string()),
        );
    }
    Ok(Some(map))
}

/// Build recorded-graph synthesis configuration when any synthesis flag is set.
fn build_synthesis(flags: &ProfileFlags) -> anyhow::Result<Option<serde_json::Value>> {
    let any = flags.synthesis_speedup_ratio.is_some()
        || flags.synthesis_prefix_len_multiplier.is_some()
        || flags.synthesis_prefix_root_multiplier.is_some()
        || flags.synthesis_prompt_len_multiplier.is_some()
        || flags.synthesis_output_len_multiplier.is_some()
        || flags.synthesis_max_isl.is_some()
        || flags.synthesis_max_osl.is_some()
        || flags.synthesis_idle_gap_cap.is_some();
    if !any {
        return Ok(None);
    }
    // clap's f64 parser accepts `nan`/`inf`; JSON has no non-finite numbers, so reject
    // them with a clean error instead of panicking in `Number::from_f64`.
    let f = |v: f64| -> anyhow::Result<serde_json::Value> {
        serde_json::Number::from_f64(v)
            .map(serde_json::Value::Number)
            .ok_or_else(|| anyhow::anyhow!("synthesis numeric flag value must be finite, got {v}"))
    };
    let mut m = serde_json::Map::new();
    m.insert(
        "speedup_ratio".into(),
        f(flags.synthesis_speedup_ratio.unwrap_or(1.0))?,
    );
    m.insert(
        "prefix_len_multiplier".into(),
        f(flags.synthesis_prefix_len_multiplier.unwrap_or(1.0))?,
    );
    m.insert(
        "prefix_root_multiplier".into(),
        serde_json::Value::from(flags.synthesis_prefix_root_multiplier.unwrap_or(1)),
    );
    m.insert(
        "prompt_len_multiplier".into(),
        f(flags.synthesis_prompt_len_multiplier.unwrap_or(1.0))?,
    );
    m.insert(
        "output_len_multiplier".into(),
        f(flags.synthesis_output_len_multiplier.unwrap_or(1.0))?,
    );
    // `max_isl`/`max_osl` are `None`-default (excluded when unset).
    if let Some(v) = flags.synthesis_max_isl {
        m.insert("max_isl".into(), serde_json::Value::from(v));
    }
    if let Some(v) = flags.synthesis_max_osl {
        m.insert("max_osl".into(), serde_json::Value::from(v));
    }
    m.insert(
        "idle_gap_cap_seconds".into(),
        f(flags.synthesis_idle_gap_cap.unwrap_or(60.0))?,
    );
    m.insert(
        "dataset_sampling_strategy".into(),
        serde_json::Value::String(
            flags
                .dataset_sampling_strategy
                .clone()
                .unwrap_or_else(|| "sequential".to_string()),
        ),
    );
    m.insert(
        "trajectory_start_min_ratio".into(),
        f(flags.trajectory_start_min_ratio.unwrap_or(0.0))?,
    );
    m.insert(
        "trajectory_start_max_ratio".into(),
        f(flags.trajectory_start_max_ratio.unwrap_or(0.0))?,
    );
    m.insert(
        "t_star_random_seed".into(),
        serde_json::Value::from(flags.random_seed.unwrap_or(0)),
    );
    Ok(Some(serde_json::Value::Object(m)))
}

/// Build accuracy configuration when `--accuracy-benchmark` is set.
fn build_accuracy(flags: &ProfileFlags) -> Option<crate::model::config::Accuracy> {
    let benchmark = flags.accuracy_benchmark.clone()?;
    let enable_cot = if flags.accuracy_enable_cot {
        Some(true)
    } else if flags.accuracy_no_enable_cot {
        Some(false)
    } else {
        None
    };
    Some(crate::model::config::Accuracy {
        benchmark,
        enable_cot,
        grader: flags.accuracy_grader.clone(),
        n_shots: flags.accuracy_n_shots,
        system_prompt: flags.accuracy_system_prompt.clone(),
        tasks: flags.accuracy_tasks.clone(),
        verbose: flags.accuracy_verbose,
    })
}

/// Build rankings distributions when any rankings flag is set.
fn build_rankings(flags: &ProfileFlags) -> Option<crate::model::dataset::Rankings> {
    let any = flags.rankings_passages_mean.is_some()
        || flags.rankings_passages_stddev.is_some()
        || flags.rankings_passages_prompt_token_mean.is_some()
        || flags.rankings_passages_prompt_token_stddev.is_some()
        || flags.rankings_query_prompt_token_mean.is_some()
        || flags.rankings_query_prompt_token_stddev.is_some();
    if !any {
        return None;
    }
    Some(crate::model::dataset::Rankings {
        passages: rankings_dist(
            flags.rankings_passages_mean,
            flags.rankings_passages_stddev,
            10.0,
        ),
        passage_tokens: rankings_dist(
            flags.rankings_passages_prompt_token_mean,
            flags.rankings_passages_prompt_token_stddev,
            128.0,
        ),
        query_tokens: rankings_dist(
            flags.rankings_query_prompt_token_mean,
            flags.rankings_query_prompt_token_stddev,
            32.0,
        ),
    })
}

/// One rankings sub-distribution: a `{mean, stddev}` normal when the mean flag is
/// set (stddev defaults to 0.0, matching `NormalDistribution`), else the config's
/// `FixedDistribution{value}` default.
fn rankings_dist(mean: Option<f64>, stddev: Option<f64>, default_value: f64) -> Distribution {
    if mean.is_none() && stddev.is_none() {
        return Distribution {
            value: Some(default_value),
            ..Default::default()
        };
    }
    Distribution {
        mean,
        stddev: Some(stddev.unwrap_or(0.0)),
        ..Default::default()
    }
}

fn build_prefix_prompts(flags: &ProfileFlags) -> Option<PrefixPrompts> {
    let any = flags.shared_system_prompt_length.is_some()
        || flags.user_context_prompt_length.is_some()
        || flags.num_prefix_prompts.is_some()
        || flags.prefix_prompt_length.is_some();
    if !any {
        return None;
    }
    Some(PrefixPrompts {
        shared_system_length: flags.shared_system_prompt_length,
        user_context_length: flags.user_context_prompt_length,
        length: flags.prefix_prompt_length,
        pool_size: flags.num_prefix_prompts,
    })
}

/// Build the synthetic audio spec when any `--audio-*` flag is set.
fn build_audio_spec(flags: &ProfileFlags) -> Option<AudioSpec> {
    let any = flags.audio_length_mean.is_some()
        || flags.audio_batch_size.is_some()
        || flags.audio_num_channels.is_some()
        || !flags.audio_depths.is_empty()
        || flags.audio_format.is_some()
        || !flags.audio_sample_rates.is_empty();
    if !any {
        return None;
    }
    let length = match flags.audio_length_mean {
        Some(mean) => Distribution {
            mean: Some(mean),
            stddev: Some(flags.audio_length_stddev.unwrap_or(0.0)),
            ..Default::default()
        },
        None => default_media_dim(),
    };
    // The wire carries sample rates in kHz (Hz / 1000).
    let sample_rates = if flags.audio_sample_rates.is_empty() {
        vec![16.0]
    } else {
        flags
            .audio_sample_rates
            .iter()
            .map(|r| r / 1000.0)
            .collect()
    };
    Some(AudioSpec {
        batch_size: implicit_media_batch(
            flags.audio_batch_size,
            flags.audio_length_mean.is_some() || flags.audio_length_stddev.is_some(),
        ),
        channels: flags.audio_num_channels.unwrap_or(1),
        depths: if flags.audio_depths.is_empty() {
            vec![16]
        } else {
            flags.audio_depths.clone()
        },
        format: flags
            .audio_format
            .clone()
            .unwrap_or_else(|| "wav".to_string()),
        length,
        sample_rates,
    })
}

/// Build the synthetic video spec when any `--video-*` flag is set.
fn build_video_spec(flags: &ProfileFlags) -> Option<VideoSpec> {
    let any = flags.video_width.is_some()
        || flags.video_height.is_some()
        || flags.video_duration.is_some()
        || flags.video_fps.is_some()
        || flags.video_format.is_some()
        || flags.video_codec.is_some()
        || flags.video_synth_type.is_some()
        || flags.video_batch_size.is_some();
    if !any {
        return None;
    }
    Some(VideoSpec {
        audio: VideoAudio {
            channels: flags.video_audio_num_channels.unwrap_or(0),
            codec: flags.video_audio_codec.clone(),
            depth: flags.video_audio_depth.unwrap_or(16),
            sample_rate: flags
                .video_audio_sample_rate
                .map(|r| r / 1000.0)
                .unwrap_or(44.1),
        },
        batch_size: implicit_media_batch(
            flags.video_batch_size,
            flags.video_width.is_some()
                || flags.video_height.is_some()
                || flags.video_duration.is_some()
                || flags.video_fps.is_some()
                || flags.video_synth_type.is_some(),
        ),
        codec: flags
            .video_codec
            .clone()
            .unwrap_or_else(|| "libvpx-vp9".to_string()),
        duration: flags.video_duration.unwrap_or(1.0),
        format: flags
            .video_format
            .clone()
            .unwrap_or_else(|| "webm".to_string()),
        fps: flags.video_fps.unwrap_or(4),
        synth_type: flags
            .video_synth_type
            .clone()
            .unwrap_or_else(|| "moving_shapes".to_string()),
        width: flags.video_width,
        height: flags.video_height,
    })
}

/// Parse `k<sep>v` pairs (e.g. `env:prod`, `svc=aiperf`) into ordered tuples.
fn parse_kv(items: &[String], sep: char) -> anyhow::Result<Vec<(String, String)>> {
    items
        .iter()
        .map(|item| {
            item.split_once(sep)
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .ok_or_else(|| anyhow::anyhow!("invalid {sep}-separated pair {item:?}"))
        })
        .collect()
}

/// Parse seconds from a number, `Ns`, `Nm`, `Nh`, or `inf`.
pub(crate) fn parse_duration_secs(s: &str) -> anyhow::Result<f64> {
    let t = s.trim();
    if t.eq_ignore_ascii_case("inf") {
        return Ok(f64::INFINITY);
    }
    let (num, mult) = if let Some(n) = t.strip_suffix(['s', 'S']) {
        (n, 1.0)
    } else if let Some(n) = t.strip_suffix(['m', 'M']) {
        (n, 60.0)
    } else if let Some(n) = t.strip_suffix(['h', 'H']) {
        (n, 3600.0)
    } else {
        (t, 1.0)
    };
    let v: f64 = num.trim().parse().map_err(|_| {
        anyhow::anyhow!("invalid duration {s:?} (use a number, '30s', '5m', '2h', or 'inf')")
    })?;
    Ok(v * mult)
}

/// Parse semicolon-separated `isl[|std],osl[|std]:prob` entries.
pub(crate) fn parse_seq_dist(s: &str) -> anyhow::Result<Vec<crate::model::dataset::SeqDistEntry>> {
    use crate::model::dataset::{Distribution, SeqDistEntry};
    let dim = |part: &str| -> anyhow::Result<Distribution> {
        let (mean, stddev) = match part.split_once('|') {
            Some((m, sd)) => (m.trim().parse::<f64>()?, sd.trim().parse::<f64>()?),
            None => (part.trim().parse::<f64>()?, 0.0),
        };
        Ok(Distribution {
            mean: Some(mean),
            stddev: Some(stddev),
            ..Default::default()
        })
    };
    s.split(';')
        .filter(|e| !e.trim().is_empty())
        .map(|entry| {
            let (pair, prob) = entry.rsplit_once(':').ok_or_else(|| {
                anyhow::anyhow!("invalid --seq-dist entry {entry:?}: missing :prob")
            })?;
            let (isl, osl) = pair.split_once(',').ok_or_else(|| {
                anyhow::anyhow!("invalid --seq-dist entry {entry:?}: missing isl,osl")
            })?;
            Ok(SeqDistEntry {
                isl: dim(isl)?,
                osl: dim(osl)?,
                probability: prob.trim().parse::<f64>()?,
            })
        })
        .collect()
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

pub(crate) fn parse_model_strategy(s: &str) -> anyhow::Result<ModelStrategy> {
    Ok(match s {
        "round_robin" => ModelStrategy::RoundRobin,
        "random" => ModelStrategy::Random,
        "weighted" => ModelStrategy::Weighted,
        other => anyhow::bail!("unknown --model-selection-strategy {other:?}"),
    })
}

/// Parse `--connection-reuse-strategy`.
pub(crate) fn parse_connection_reuse(s: &str) -> anyhow::Result<ConnectionReuse> {
    Ok(match s {
        "pooled" => ConnectionReuse::Pooled,
        "never" => ConnectionReuse::Never,
        "sticky-user-sessions" => ConnectionReuse::StickyUserSessions,
        other => anyhow::bail!("unknown --connection-reuse-strategy {other:?}"),
    })
}

/// Parse `--request-content-type` (MIME string) into the wire token.
pub(crate) fn parse_content_type(s: &str) -> anyhow::Result<RequestContentType> {
    Ok(match s {
        "application/json" => RequestContentType::ApplicationJson,
        "multipart/form-data" => RequestContentType::MultipartFormData,
        other => anyhow::bail!("unknown --request-content-type {other:?}"),
    })
}

/// Parse `--wait-for-model-mode`.
pub(crate) fn parse_wait_mode(s: &str) -> anyhow::Result<WaitForModelMode> {
    Ok(match s {
        "models" => WaitForModelMode::Models,
        "inference" => WaitForModelMode::Inference,
        "both" => WaitForModelMode::Both,
        other => anyhow::bail!("unknown --wait-for-model-mode {other:?}"),
    })
}

/// A linear ramp of the given duration (the default ramp strategy).
fn linear_ramp(duration: f64) -> crate::model::phase::Ramp {
    crate::model::phase::Ramp {
        duration,
        strategy: "linear".to_string(),
    }
}

/// Count the non-empty lines of a fixed-schedule input (its entry count). A
/// directory input (e.g. a SageMaker capture dir) is recursed for `*.jsonl` and
/// the non-empty lines summed across files — the same set the loader reads.
fn count_schedule_entries(path: &std::path::Path) -> anyhow::Result<u64> {
    if path.is_dir() {
        let mut total = 0u64;
        let mut stack = vec![path.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).map_err(|e| {
                anyhow::anyhow!("failed to read schedule dir {}: {e}", dir.display())
            })? {
                let p = entry
                    .map_err(|e| anyhow::anyhow!("failed to read schedule dir entry: {e}"))?
                    .path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|e| e == "jsonl") {
                    let text = std::fs::read_to_string(&p).map_err(|e| {
                        anyhow::anyhow!("failed to read schedule {}: {e}", p.display())
                    })?;
                    total += text.lines().filter(|l| !l.trim().is_empty()).count() as u64;
                }
            }
        }
        return Ok(total);
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read schedule {}: {e}", path.display()))?;
    Ok(text.lines().filter(|l| !l.trim().is_empty()).count() as u64)
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

/// Reject a comma-list value on a sweep-capable flag (multi-run is deferred).
fn reject_sweep(flag: &str, value: Option<&str>) -> anyhow::Result<()> {
    if let Some(v) = value
        && v.contains(',')
    {
        anyhow::bail!("`{flag} {v}` describes a sweep but reached the single-run resolver");
    }
    Ok(())
}

/// Parse a single scalar from a sweep-capable flag (comma-lists already rejected).
pub(crate) fn parse_single<T: std::str::FromStr>(
    flag: &str,
    value: Option<&str>,
) -> anyhow::Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    match value {
        None => Ok(None),
        Some(v) => v
            .parse::<T>()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("invalid value for {flag}: {v} ({e})")),
    }
}

#[cfg(test)]
mod tests {
    use super::{is_fake_model_name, is_truthy_env};

    #[test]
    fn synthesis_rejects_non_finite_value() {
        use crate::flags::ProfileFlags;
        use clap::Parser;

        for bad in ["nan", "inf", "-inf"] {
            let flags =
                ProfileFlags::try_parse_from(["profile", &format!("--synthesis-speedup-ratio={bad}")])
                    .expect("flags parse");
            let err = super::build_synthesis(&flags)
                .expect_err("non-finite synthesis value must be a clean error, not a panic");
            assert!(
                err.to_string().contains("finite"),
                "expected a finiteness error for {bad:?}, got: {err}"
            );
        }
    }

    #[test]
    fn synthesis_accepts_finite_values() {
        use crate::flags::ProfileFlags;
        use clap::Parser;

        let flags = ProfileFlags::try_parse_from(["profile", "--synthesis-speedup-ratio", "2.5"])
            .expect("flags parse");
        let value = super::build_synthesis(&flags)
            .expect("finite value builds")
            .expect("synthesis flag set yields Some");
        assert_eq!(value["speedup_ratio"], serde_json::json!(2.5));
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

    /// `--dry-run` projects the dataset-analysis artifact toggle and threads the
    /// `--kv-*` knobs into `artifacts.dataset_analysis_*`; `--no-dataset-analysis`
    /// clears it and a real (non-dry-run) run never carries it.
    #[test]
    fn dry_run_projects_dataset_analysis() {
        // `resolve` builds the full (large) BenchmarkConfig on the stack, which
        // overflows the default test-thread stack; run on a generous one.
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(dry_run_projects_dataset_analysis_body)
            .expect("spawn worker")
            .join()
            .expect("worker panicked");
    }

    fn dry_run_projects_dataset_analysis_body() {
        use crate::flags::ProfileFlags;

        let project = |args: &[&str]| {
            let flags = ProfileFlags::parse_from_args(
                &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            )
            .expect("parse flags");
            super::resolve(&flags)
                .expect("resolve run")
                .cfg
                .artifacts
                .expect("artifacts present")
        };

        // Dry-run emits the analysis with the KV knobs threaded through.
        let a = project(&[
            "-m",
            "mock-model",
            "--endpoint-type",
            "chat",
            "--dry-run",
            "--kv-block-size",
            "32",
        ]);
        assert_eq!(
            a.dataset_analysis_path.as_deref(),
            Some("dataset_analysis.json")
        );
        assert_eq!(a.dataset_analysis_block_size, Some(32));

        // `--kv-cache-blocks` + per-conversation project through.
        let a = project(&[
            "-m",
            "mock-model",
            "--endpoint-type",
            "chat",
            "--dry-run",
            "--kv-cache-blocks",
            "128",
            "--dataset-analysis-per-conversation",
        ]);
        assert_eq!(a.dataset_analysis_cache_blocks, Some(128));
        assert!(a.dataset_analysis_per_conversation);

        // Suppression clears the toggle even under `--dry-run`.
        let a = project(&[
            "-m",
            "mock-model",
            "--endpoint-type",
            "chat",
            "--dry-run",
            "--no-dataset-analysis",
        ]);
        assert!(a.dataset_analysis_path.is_none());

        // A real run never carries the analysis.
        let a = project(&[
            "-m",
            "mock-model",
            "--endpoint-type",
            "chat",
            "-u",
            "http://localhost:8000",
        ]);
        assert!(a.dataset_analysis_path.is_none());
    }

    fn parse(args: &[&str]) -> crate::flags::ProfileFlags {
        crate::flags::ProfileFlags::parse_from_args(
            &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        )
        .expect("parse flags")
    }

    /// `resolve` builds the full (large) BenchmarkConfig on the stack, which
    /// overflows the default test-thread stack; run on a generous one.
    fn run_on_big_stack(body: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(body)
            .expect("spawn worker")
            .join()
            .expect("worker panicked");
    }

    #[test]
    fn baseten_only_flags_rejected_without_baseten_trace() {
        run_on_big_stack(baseten_only_flags_rejected_without_baseten_trace_body);
    }

    fn baseten_only_flags_rejected_without_baseten_trace_body() {
        // Wrong --custom-dataset-type.
        let flags = parse(&[
            "-m",
            "mock-model",
            "--endpoint-type",
            "chat",
            "-u",
            "http://localhost:8000",
            "--input-file",
            "trace.jsonl",
            "--custom-dataset-type",
            "mooncake_trace",
            "--replay-speedup",
            "2.0",
        ]);
        let error = super::resolve(&flags).unwrap_err();
        assert!(error.to_string().contains("baseten_trace loader"));
        assert!(error.to_string().contains("mooncake_trace"));

        // No --input-file at all (synthetic dataset).
        let flags = parse(&[
            "-m",
            "mock-model",
            "--endpoint-type",
            "chat",
            "-u",
            "http://localhost:8000",
            "--open-loop-strict",
        ]);
        let error = super::resolve(&flags).unwrap_err();
        assert!(error.to_string().contains("baseten_trace loader"));
        assert!(error.to_string().contains("--input-file"));
    }

    #[test]
    fn baseten_only_flags_accepted_with_baseten_trace() {
        run_on_big_stack(baseten_only_flags_accepted_with_baseten_trace_body);
    }

    fn baseten_only_flags_accepted_with_baseten_trace_body() {
        let flags = parse(&[
            "-m",
            "mock-model",
            "--endpoint-type",
            "chat",
            "-u",
            "http://localhost:8000",
            "--input-file",
            "trace.parquet",
            "--custom-dataset-type",
            "baseten_trace",
            "--replay-speedup",
            "2.0",
            "--omit-kv-hints",
        ]);
        // Validation passes; may still fail later for unrelated reasons (no
        // real file on disk), but never with a baseten-only-flags message.
        if let Err(error) = super::resolve(&flags) {
            assert!(!error.to_string().contains("baseten_trace loader"));
        }
    }

    #[test]
    fn prompt_corpus_flag_projects_synthetic_dataset() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--prompt-corpus",
                "coding",
            ]);
            let run = super::resolve(&flags).expect("resolve run");
            let value = serde_json::to_value(&run).expect("serialize run");
            assert_eq!(
                value["cfg"]["datasets"][0]["prompts"]["corpus"],
                serde_json::json!("coding")
            );
        });
    }

    #[test]
    fn prompt_corpus_flag_projects_file_dataset_prompts() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--input-file",
                "trace.jsonl",
                "--custom-dataset-type",
                "mooncake_trace",
                "--prompt-corpus",
                "random",
            ]);
            let run = super::resolve(&flags).expect("resolve run");
            let value = serde_json::to_value(&run).expect("serialize run");
            assert_eq!(
                value["cfg"]["datasets"][0]["type"],
                serde_json::json!("file")
            );
            assert_eq!(
                value["cfg"]["datasets"][0]["prompts"]["corpus"],
                serde_json::json!("random")
            );
            assert_eq!(
                value["cfg"]["datasets"][0]["synthesis"]["corpus"],
                serde_json::Value::Null
            );
        });
    }

    #[test]
    fn prompt_corpus_flag_projects_public_dataset_prompts() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--public-dataset",
                "sharegpt",
                "--prompt-corpus",
                "coding",
            ]);
            let run = super::resolve(&flags).expect("resolve run");
            let value = serde_json::to_value(&run).expect("serialize run");
            assert_eq!(
                value["cfg"]["datasets"][0]["type"],
                serde_json::json!("public")
            );
            assert_eq!(
                value["cfg"]["datasets"][0]["prompts"]["corpus"],
                serde_json::json!("coding")
            );
        });
    }

    #[test]
    fn proxy_flag_projects_onto_endpoint() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "-u",
                "http://remote:8000",
                "--proxy",
                "http://user:pass@proxy:3128",
            ]);
            let run = super::resolve(&flags).expect("resolve run");
            let value = serde_json::to_value(&run).expect("serialize run");
            assert_eq!(
                value["cfg"]["endpoint"]["proxy"],
                serde_json::json!("http://user:pass@proxy:3128")
            );
        });
    }

    #[test]
    fn proxy_from_env_flag_projects_onto_endpoint() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "-u",
                "http://remote:8000",
                "--proxy-from-env",
            ]);
            let run = super::resolve(&flags).expect("resolve run");
            let value = serde_json::to_value(&run).expect("serialize run");
            assert_eq!(
                value["cfg"]["endpoint"]["proxy_from_env"],
                serde_json::json!(true)
            );
        });
    }

    #[test]
    fn hf_dataset_flag_builds_public_without_catalog() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--hf-dataset",
                "allenai/WildChat",
                "--hf-split",
                "train",
                "--hf-output-len",
                "128",
            ]);
            let run = super::resolve(&flags).expect("resolve run");
            let value = serde_json::to_value(&run).expect("serialize run");
            let ds = &value["cfg"]["datasets"][0];
            assert_eq!(ds["type"], serde_json::json!("public"));
            assert_eq!(ds["name"], serde_json::json!("allenai/WildChat"));
            assert_eq!(ds["format"], serde_json::json!("hf"));
            assert_eq!(ds["source"]["type"], serde_json::json!("hugging_face"));
            assert_eq!(
                ds["source"]["dataset"],
                serde_json::json!("allenai/WildChat")
            );
            assert_eq!(ds["source"]["split"], serde_json::json!("train"));
            assert_eq!(ds["options"]["output_len"], serde_json::json!(128));
        });
    }

    #[test]
    fn hf_format_flag_overrides_projected_format_and_bypasses_catalog() {
        run_on_big_stack(|| {
            // `--hf-format` forces a specific registered loader for `--hf-dataset`
            // instead of the auto-detecting `hf` format, while still projecting a
            // `public` dataset whose source is the arbitrary repo id (no catalog
            // lookup). Column overrides land in the loader options bag.
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--hf-dataset",
                "some-org/convo-set",
                "--hf-format",
                "hf_conversation",
                "--hf-subset",
                "default",
                "--hf-text-column",
                "conversations",
                "--hf-output-column",
                "reply",
            ]);
            let run = super::resolve(&flags).expect("resolve run");
            let value = serde_json::to_value(&run).expect("serialize run");
            let ds = &value["cfg"]["datasets"][0];
            assert_eq!(ds["type"], serde_json::json!("public"));
            assert_eq!(ds["name"], serde_json::json!("some-org/convo-set"));
            // The override wins over the default `hf` format.
            assert_eq!(ds["format"], serde_json::json!("hf_conversation"));
            assert_eq!(ds["source"]["type"], serde_json::json!("hugging_face"));
            assert_eq!(
                ds["source"]["dataset"],
                serde_json::json!("some-org/convo-set")
            );
            assert_eq!(ds["source"]["subset"], serde_json::json!("default"));
            assert_eq!(
                ds["options"]["text_column"],
                serde_json::json!("conversations")
            );
            assert_eq!(ds["options"]["output_column"], serde_json::json!("reply"));
        });
    }

    #[test]
    fn baseten_extra_input_collisions_are_rejected_and_opt_outable() {
        run_on_big_stack(baseten_extra_input_collisions_are_rejected_and_opt_outable_body);
    }

    fn baseten_extra_input_collisions_are_rejected_and_opt_outable_body() {
        let flags = parse(&[
            "-m",
            "mock-model",
            "--endpoint-type",
            "chat",
            "-u",
            "http://localhost:8000",
            "--input-file",
            "trace.parquet",
            "--custom-dataset-type",
            "baseten_trace",
            "--extra-inputs",
            "min_tokens:5",
            "--extra-inputs",
            "hash_ids:1",
        ]);
        let error = super::resolve(&flags).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("min_tokens"));
        assert!(message.contains("--no-force-min-tokens"));
        assert!(message.contains("hash_ids"));
        assert!(message.contains("--omit-kv-hints"));

        // The opt-out flags let the same extras through.
        let flags = parse(&[
            "-m",
            "mock-model",
            "--endpoint-type",
            "chat",
            "-u",
            "http://localhost:8000",
            "--input-file",
            "trace.parquet",
            "--custom-dataset-type",
            "baseten_trace",
            "--extra-inputs",
            "min_tokens:5",
            "--extra-inputs",
            "hash_ids:1",
            "--no-force-min-tokens",
            "--omit-kv-hints",
        ]);
        if let Err(error) = super::resolve(&flags) {
            assert!(!error.to_string().contains("overwritten per-turn"));
        }
    }

    #[test]
    fn endpoint_control_hook_flags_project_endpoint_overrides() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--reset-kv-cache-timeout-seconds",
                "3.5",
                "--reset-kv-cache-path",
                "/reset_prefix_cache",
                "--server-profiler-timeout-seconds",
                "10",
                "--server-profiler-start-path",
                "/start_profile",
                "--server-profiler-stop-path",
                "/stop_profile",
            ]);
            let run = super::resolve(&flags).expect("resolve run");
            let endpoint = run.cfg.endpoint.expect("endpoint present");
            let reset_kv_cache = endpoint.reset_kv_cache.expect("reset_kv_cache enabled");
            assert_eq!(reset_kv_cache.timeout_seconds, Some(3.5));
            assert_eq!(reset_kv_cache.path.as_deref(), Some("/reset_prefix_cache"));
            let server_profiler = endpoint.server_profiler.expect("server_profiler enabled");
            assert_eq!(server_profiler.timeout_seconds, Some(10.0));
            assert_eq!(
                server_profiler.start_path.as_deref(),
                Some("/start_profile")
            );
            assert_eq!(server_profiler.stop_path.as_deref(), Some("/stop_profile"));
        });
    }

    /// Without `--scenario`, resolution leaves `resolved.scenario_outcome` unset.
    #[cfg(feature = "agentx")]
    #[test]
    fn no_scenario_leaves_outcome_unset() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "-u",
                "http://localhost:8000",
            ]);
            let run = super::resolve(&flags).expect("resolve run");
            assert!(run.resolved.scenario_outcome.is_none());
        });
    }

    /// `--scenario inferencex-agentx-mvp` on the synthetic-default dataset is a
    /// non-overridable lock failure (`require_loader` forbids the synthetic
    /// default), so resolution fails — proving the scenario resolver is wired
    /// into the CLI Config-v2 pipeline.
    #[cfg(feature = "agentx")]
    #[test]
    fn scenario_hard_fails_on_synthetic_default_dataset() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "-u",
                "http://localhost:8000",
                "--streaming",
                "--scenario",
                "inferencex-agentx-mvp",
            ]);
            let err = super::resolve(&flags).expect_err("scenario lock must fail");
            assert!(
                err.to_string().contains("scenario lock failure"),
                "unexpected error: {err}"
            );
        });
    }

    /// `--agentic-cache-warmup-duration` on a non-agentic run (no scenario / no
    /// `--weka-semantics legacy`) is rejected: the accelerated cache-warmup
    /// substage would be silently dropped, so the flag is an invisible no-op.
    #[test]
    fn agentic_cache_warmup_rejected_without_agentic_replay() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "-u",
                "http://localhost:8000",
                "--agentic-cache-warmup-duration",
                "5",
            ]);
            let err = super::resolve(&flags).expect_err("guard must reject non-agentic run");
            assert!(
                err.to_string()
                    .contains("--agentic-cache-warmup-duration requires the agentic_replay"),
                "unexpected error: {err}"
            );
        });
    }

    /// Under `--weka-semantics legacy` the run resolves to the agentic_replay
    /// timing mode, so `--agentic-cache-warmup-duration` passes the guard (any
    /// failure must not be the guard's own rejection).
    #[test]
    fn agentic_cache_warmup_accepted_under_legacy_weka() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "-u",
                "http://localhost:8000",
                "--streaming",
                "--weka-semantics",
                "legacy",
                "--agentic-cache-warmup-duration",
                "5",
            ]);
            if let Err(err) = super::resolve(&flags) {
                assert!(
                    !err.to_string()
                        .contains("--agentic-cache-warmup-duration requires the agentic_replay"),
                    "guard must not fire under legacy weka: {err}"
                );
            }
        });
    }

    /// An unknown `--scenario` name is rejected during resolution.
    #[cfg(feature = "agentx")]
    #[test]
    fn unknown_scenario_rejected() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "-u",
                "http://localhost:8000",
                "--scenario",
                "does-not-exist",
            ]);
            let err = super::resolve(&flags).expect_err("unknown scenario must fail");
            assert!(err.to_string().contains("unknown --scenario"), "got: {err}");
        });
    }
}
