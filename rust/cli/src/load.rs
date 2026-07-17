// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Parse `profile` flags or YAML config into a [`BenchmarkRun`].
//!
//! Flag and YAML inputs normalize to [`Inputs`] and share [`build`], keeping wire
//! defaults in one place.

use std::path::PathBuf;

use crate::flags::ProfileFlags;
use crate::model::artifacts::Artifacts;
use crate::model::dataset::{
    AudioSpec, Dataset, Distribution, ImageSpec, PrefixPrompts, Prompts, Sampling, Synthetic,
    VideoAudio, VideoSpec,
};
use crate::model::endpoint::{
    ConnectionReuse, Endpoint, EndpointType, RequestContentType, WaitForModelMode,
};
use crate::model::metrics::Metrics;
use crate::model::models::{ModelItem, ModelStrategy, Models};
use crate::model::phase::{AdaptiveScale, Phase, PhaseCommon, PhaseKind, SlaFilter};
use crate::model::runtime::Runtime;
use crate::model::tokenizer::Tokenizer;
use crate::model::transport::Transport;
use crate::model::{BenchmarkConfig, BenchmarkRun, Resolved};

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
    /// Custom request path appended to the endpoint URL (`endpoint.path`).
    pub endpoint_path: Option<String>,
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
    /// Inter-turn delay cap, seconds (file datasets).
    pub inter_turn_delay_cap_seconds: Option<f64>,
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
    /// Bounded-memory sketch metric retention.
    pub sketch_metrics: bool,
    /// Synthetic image spec (present when any image flag is set).
    pub image_spec: Option<ImageSpec>,
    /// Synthetic audio spec.
    pub audio_spec: Option<AudioSpec>,
    /// Synthetic video spec.
    pub video_spec: Option<VideoSpec>,
    /// Adaptive-scale controller (present when --adaptive-scale is set).
    pub adaptive_scale: Option<AdaptiveScale>,
    /// Shared-prefix / prefix-pool policy.
    pub prefix_prompts: Option<PrefixPrompts>,
    pub artifact_dir: PathBuf,
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
        endpoint_path: flags.custom_endpoint.clone(),
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
        user_centric: match (flags.user_centric_rate, flags.num_users) {
            (Some(rate), Some(users)) => Some((rate, users)),
            _ => None,
        },
        request_count,
        benchmark_duration,
        grace_period: flags.benchmark_grace_period,
        warmup,
        random_seed: flags.random_seed,
        dataset_random_seed: flags.random_seed,
        runtime_workers: None,
        runtime_workers_min: None,
        runtime_cells: flags.cells.unwrap_or(1),
        input_file: flags.input_file.clone(),
        inline_records: None,
        custom_dataset_type: flags.custom_dataset_type.clone(),
        public_dataset: flags.public_dataset.clone(),
        hf_subset: flags.hf_subset.clone(),
        inter_turn_delay_cap_seconds: flags.inter_turn_delay_cap_seconds,
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
        sketch_metrics: flags.sketch_metrics,
        image_spec: build_image_spec(flags),
        audio_spec: build_audio_spec(flags),
        video_spec: build_video_spec(flags),
        adaptive_scale: build_adaptive_scale(flags, concurrency)?,
        prefix_prompts: build_prefix_prompts(flags),
        scenario: flags.scenario.clone(),
        trajectory_start_min_ratio: flags.trajectory_start_min_ratio.unwrap_or(0.0),
        trajectory_start_max_ratio: flags.trajectory_start_max_ratio.unwrap_or(0.0),
        unsafe_override: flags.unsafe_override,
        agentic_cache_warmup_duration: flags.agentic_cache_warmup_duration,
        rankings: build_rankings(flags),
        accuracy: build_accuracy(flags),
        synthesis: build_synthesis(flags),
        dataset_filters: parse_dataset_filters(flags)?,
        artifact_dir: flags
            .artifact_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("artifacts")),
    };
    build(inputs)
}

/// Build one run from normalized inputs.
pub(crate) fn build(inputs: Inputs) -> anyhow::Result<BenchmarkRun> {
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
    };

    let dataset = if let Some(name) = &inputs.public_dataset {
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
        })
    } else if inputs.input_file.is_some() || inputs.inline_records.is_some() {
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
            records: inputs.inline_records.clone(),
            synthesis: inputs.synthesis.clone(),
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
                sequence_distribution: inputs.sequence_distribution.clone(),
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
            },
            kind: PhaseKind::FixedSchedule {
                auto_offset,
                start_offset: inputs.fixed_schedule_start_offset,
                end_offset: inputs.fixed_schedule_end_offset,
            },
        }
    } else {
        let mut phase = build_phase(
            "profiling",
            false,
            inputs.concurrency.unwrap_or(1),
            inputs.request_rate,
            inputs.rate_mode.as_deref(),
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
        phase
    };
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
        // A cache-warmup duration requires a phase even when no warmup is authored.
        let mut wp = build_phase(
            "warmup", true, 1, None, None, None, None, None, None, None, None,
        );
        wp.common.agentic_cache_warmup_duration = Some(dur);
        phases.push(wp);
    }
    phases.push(profiling);

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
        }),
        metrics: Some(Metrics {
            slos: inputs.slos.clone(),
            slice_duration_seconds: inputs.slice_duration,
            sketch: sketch_metrics.then_some(true),
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
        resolved: Resolved::default(),
        variables: serde_json::Map::new(),
    })
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
    Phase {
        common: PhaseCommon {
            name: name.to_string(),
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
        },
        kind,
    }
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
        flags.public_dataset.is_some(),
        "--dataset-filter requires --public-dataset"
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
fn build_synthesis(flags: &ProfileFlags) -> Option<serde_json::Value> {
    let any = flags.synthesis_speedup_ratio.is_some()
        || flags.synthesis_prefix_len_multiplier.is_some()
        || flags.synthesis_prefix_root_multiplier.is_some()
        || flags.synthesis_prompt_len_multiplier.is_some()
        || flags.synthesis_output_len_multiplier.is_some()
        || flags.synthesis_max_isl.is_some()
        || flags.synthesis_max_osl.is_some()
        || flags.synthesis_idle_gap_cap.is_some();
    if !any {
        return None;
    }
    let f = |v: f64| {
        serde_json::Number::from_f64(v)
            .map(serde_json::Value::Number)
            .unwrap()
    };
    let mut m = serde_json::Map::new();
    m.insert(
        "speedup_ratio".into(),
        f(flags.synthesis_speedup_ratio.unwrap_or(1.0)),
    );
    m.insert(
        "prefix_len_multiplier".into(),
        f(flags.synthesis_prefix_len_multiplier.unwrap_or(1.0)),
    );
    m.insert(
        "prefix_root_multiplier".into(),
        serde_json::Value::from(flags.synthesis_prefix_root_multiplier.unwrap_or(1)),
    );
    m.insert(
        "prompt_len_multiplier".into(),
        f(flags.synthesis_prompt_len_multiplier.unwrap_or(1.0)),
    );
    m.insert(
        "output_len_multiplier".into(),
        f(flags.synthesis_output_len_multiplier.unwrap_or(1.0)),
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
        f(flags.synthesis_idle_gap_cap.unwrap_or(60.0)),
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
        f(flags.trajectory_start_min_ratio.unwrap_or(0.0)),
    );
    m.insert(
        "trajectory_start_max_ratio".into(),
        f(flags.trajectory_start_max_ratio.unwrap_or(0.0)),
    );
    m.insert(
        "t_star_random_seed".into(),
        serde_json::Value::from(flags.random_seed.unwrap_or(0)),
    );
    Some(serde_json::Value::Object(m))
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
fn parse_single<T: std::str::FromStr>(flag: &str, value: Option<&str>) -> anyhow::Result<Option<T>>
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
