// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Pre-translation: parse `profile` flags or a YAML config into the native
//! [`BenchmarkRun`].
//!
//! This is the reverse of Python's `rust_wire` projection: instead of lowering a
//! domain config into a wire dict, it builds the one native object (which *is*
//! the wire request) directly from the input surface. Both surfaces normalize to
//! [`Inputs`] and share one [`build`] core, so the wire defaults live in exactly
//! one place. The flag surface lives here; the YAML surface lives in
//! [`crate::yaml`].
//!
//! Defaults are the Python single-run synthetic defaults, proven byte-exact
//! against the golden vectors in `tools/parity/golden/`.

use std::path::PathBuf;

use crate::flags::ProfileFlags;
use crate::model::artifacts::Artifacts;
use crate::model::dataset::{Dataset, Distribution, ImageSpec, Prompts, Sampling, Synthetic};
use crate::model::endpoint::{
    ConnectionReuse, Endpoint, EndpointType, RequestContentType, WaitForModelMode,
};
use crate::model::metrics::Metrics;
use crate::model::models::{ModelItem, ModelStrategy, Models};
use crate::model::phase::{Phase, PhaseCommon, PhaseKind};
use crate::model::runtime::Runtime;
use crate::model::tokenizer::Tokenizer;
use crate::model::transport::Transport;
use crate::model::{BenchmarkConfig, BenchmarkRun, Resolved};

// Python single-run synthetic defaults (see `src/aiperf/config/endpoint.py`,
// `dataset/*`), proven against the goldens.
const DEFAULT_TIMEOUT_SECONDS: f64 = 21600.0;
const DEFAULT_CONNECTION_LIMIT: u32 = 2500;
const DEFAULT_KEEPALIVE_TIMEOUT: f64 = 300.0;
const DEFAULT_WAIT_FOR_MODEL_INTERVAL: f64 = 5.0;
const DEFAULT_ISL_MEAN: f64 = 550.0;
/// Default synthetic conversation count when no request bound is given.
pub(crate) const DEFAULT_ENTRIES: u32 = 100;

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
    /// Warmup concurrency-ramp duration.
    pub concurrency_ramp: Option<f64>,
    /// Warmup rate-ramp duration.
    pub rate_ramp: Option<f64>,
    /// Warmup duration bound.
    pub duration: Option<f64>,
    /// Warmup grace period.
    pub grace_period: Option<f64>,
}

/// Normalized inputs both surfaces (flags / YAML) resolve to before building.
pub(crate) struct Inputs {
    pub model_names: Vec<String>,
    pub urls: Vec<String>,
    pub endpoint_type: String,
    pub transport: crate::model::transport::Transport,
    pub streaming: bool,
    pub timeout_seconds: Option<f64>,
    pub use_legacy_max_tokens: bool,
    pub use_server_token_count: bool,
    pub connection_reuse: Option<ConnectionReuse>,
    pub request_content_type: Option<RequestContentType>,
    pub wait_for_model_timeout: Option<f64>,
    pub wait_for_model_mode: Option<WaitForModelMode>,
    pub wait_for_model_interval: Option<f64>,
    pub apply_chat_template: bool,
    pub prefill_concurrency: Option<u32>,
    pub prefill_ramp: Option<f64>,
    pub gpu_telemetry_enabled: bool,
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
    pub batch_size: u32,
    pub sampling: String,
    pub entries: u32,
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
    pub random_seed: Option<u64>,
    /// File-backed dataset path (mutually exclusive with the synthetic path).
    pub input_file: Option<PathBuf>,
    /// File dataset format id (`--custom-dataset-type`).
    pub custom_dataset_type: Option<String>,
    /// Named public dataset (mutually exclusive with synthetic/file).
    pub public_dataset: Option<String>,
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

/// Resolve `profile` flags into one native run, or a clear error.
///
/// Rejects multi-run (any comma-list sweep axis) since multi-run/orchestration
/// is deferred.
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
    anyhow::ensure!(!flags.urls.is_empty(), "at least one --url is required");
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
            concurrency_ramp: flags.warmup_concurrency_ramp_duration,
            rate_ramp: flags.warmup_request_rate_ramp_duration,
            duration: flags.warmup_duration,
            grace_period: flags.warmup_grace_period,
        })
    };

    let inputs = Inputs {
        model_names: flags.model_names.clone(),
        urls: flags.urls.clone(),
        endpoint_type,
        transport: Transport::Http,
        streaming: flags.streaming,
        timeout_seconds: flags.request_timeout_seconds,
        use_legacy_max_tokens: flags.use_legacy_max_tokens,
        use_server_token_count: flags.use_server_token_count,
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
        server_metrics_enabled: !flags.no_server_metrics,
        server_metrics_formats: (!flags.server_metrics_formats.is_empty())
            .then(|| flags.server_metrics_formats.clone()),
        slos: parse_goodput(flags.goodput.as_deref())?,
        network_latency_mean: flags.network_latency_mean,
        network_latency_probe: flags
            .network_latency_automatic
            .then(|| flags.network_latency_ping_interval.unwrap_or(1.0)),
        otel_url: flags.otel_url.clone(),
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
        batch_size: 1,
        sampling: "sequential".to_string(),
        entries: num_conversations
            .or(num_sessions)
            .or(request_count.map(|n| n as u32))
            .unwrap_or(DEFAULT_ENTRIES),
        sessions: num_conversations.or(num_sessions).map(u64::from),
        concurrency,
        request_rate,
        rate_mode: flags.request_rate_mode.clone(),
        smoothness: flags.arrival_smoothness,
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
        input_file: flags.input_file.clone(),
        custom_dataset_type: flags.custom_dataset_type.clone(),
        public_dataset: flags.public_dataset.clone(),
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
        artifact_dir: flags
            .artifact_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("artifacts")),
    };
    build(inputs)
}

/// Build the one native run from normalized inputs. This is the single place the
/// wire defaults live; both the flag and YAML surfaces funnel through here.
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
        download_video_content: false,
        extra: serde_json::Map::new(),
        headers: inputs.headers,
        http2: false,
        wait_for_model_timeout: inputs.wait_for_model_timeout.unwrap_or(0.0),
        wait_for_model_interval: inputs
            .wait_for_model_interval
            .unwrap_or(DEFAULT_WAIT_FOR_MODEL_INTERVAL),
        wait_for_model_mode: inputs
            .wait_for_model_mode
            .unwrap_or(WaitForModelMode::Inference),
        path: None,
        api_key: inputs.api_key,
        session_header: inputs.session_header,
        request_content_type: inputs.request_content_type,
        template: None,
        response_field: None,
    };

    let tokenizer = Tokenizer {
        name: inputs
            .tokenizer_name
            .unwrap_or_else(|| primary_model.clone()),
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
        if let Some(max) = crate::model::public_catalog::max_conversations(
            meta,
            Some(inputs.entries),
            inputs.request_count,
        ) {
            options.insert("max_conversations".to_string(), serde_json::json!(max));
        }
        Dataset::Public(crate::model::dataset::PublicDataset {
            name: name.clone(),
            format: meta.format.clone(),
            source: meta.source.clone(),
            options,
            sampling: Sampling(inputs.sampling.clone()),
            entries: Some(inputs.entries),
            random_seed: inputs.random_seed,
        })
    } else if let Some(path) = &inputs.input_file {
        Dataset::File(crate::model::dataset::FileDataset {
            format: inputs
                .custom_dataset_type
                .clone()
                .unwrap_or_else(|| "single_turn".to_string()),
            sampling: Sampling(inputs.sampling.clone()),
            options: serde_json::Map::new(),
            path: Some(absolute_path(path)),
            // Fixed-schedule derives the count into the phase's `requests`, not
            // the dataset's `entries` (which stays unset).
            entries: if inputs.fixed_schedule.is_some() {
                None
            } else {
                Some(inputs.entries)
            },
            random_seed: inputs.random_seed,
            osl: inputs.osl.clone(),
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
            },
            images: inputs.image_spec.clone(),
            sampling: Sampling(inputs.sampling.clone()),
            turns: inputs.turns.clone(),
            turn_delay_ratio: inputs.turn_delay_ratio,
            entries: Some(inputs.entries),
            num_conversations: None,
            turn_delay_ms: inputs.turn_delay_ms.clone(),
        })
    };

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
            inputs.request_count,
            inputs.sessions,
            inputs.benchmark_duration,
            inputs.grace_period,
        );
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
            None,
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
        phases.push(wp);
    }
    phases.push(profiling);

    let endpoint_type = endpoint.endpoint_type.0.clone();
    let endpoint_urls = endpoint.urls.clone();
    // DynoSim co-simulation opens no sockets, so every online sidecar is forced
    // off (mirrors `_authored_sidecars`); other transports keep the default
    // GPU-telemetry + server-metrics scraping.
    let is_dynosim = matches!(
        inputs.transport,
        Transport::DynosimOffline | Transport::DynosimOnline
    );
    // DynoSim forces all sidecars off; otherwise GPU-telemetry and
    // server-metrics scraping are enabled by default and independently toggled.
    let gpu_enabled = inputs.gpu_telemetry_enabled && !is_dynosim;
    let server_enabled = inputs.server_metrics_enabled && !is_dynosim;
    // Network-latency calibration: fixed mean or automatic probe (disabled by
    // default). Lowered into a sidecar mirroring `_network_latency`.
    let mut network_latency_cfg = crate::model::telemetry::NetworkLatencyConfig::default();
    let network_latency_sidecar = if is_dynosim {
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
        gpu_telemetry: gpu_enabled.then(crate::model::telemetry::GpuTelemetrySidecar::default_dcgm),
        server_metrics: server_enabled.then(|| {
            let sc =
                crate::model::telemetry::ServerMetricsSidecar::from_endpoint_urls(&endpoint_urls);
            match &inputs.server_metrics_formats {
                Some(formats) => sc.with_formats(formats.clone()),
                None => sc,
            }
        }),
        network_latency: network_latency_sidecar,
    };
    let mut gpu_cfg = crate::model::telemetry::GpuTelemetryConfig::default();
    gpu_cfg.enabled = inputs.gpu_telemetry_enabled;
    let mut server_cfg = crate::model::telemetry::ServerMetricsConfig::default();
    server_cfg.enabled = inputs.server_metrics_enabled;
    if let Some(formats) = &inputs.server_metrics_formats {
        server_cfg.formats = formats.clone();
    }
    let mut cfg = BenchmarkConfig {
        models: Some(models),
        endpoint: Some(endpoint),
        tokenizer: Some(tokenizer),
        transport: Some(inputs.transport),
        runtime: Some(Runtime::default()),
        metrics: Some(Metrics {
            slos: inputs.slos.clone(),
            slice_duration_seconds: inputs.slice_duration,
            sketch: inputs.sketch_metrics.then_some(true),
        }),
        slos: (!inputs.slos.is_empty()).then(|| inputs.slos.clone()),
        artifacts: Some(Artifacts {
            trace: false,
            inputs_path: "inputs.json".to_string(),
            // Sketch retention keeps no per-record values, so the per-record
            // JSONL is dropped (mirrors `_authored_artifacts`).
            records_path: (!inputs.sketch_metrics).then(|| "profile_export.jsonl".to_string()),
            ..Default::default()
        }),
        datasets: Some(vec![dataset]),
        phases: Some(phases),
        export: None,
        gpu_telemetry: Some(gpu_cfg),
        server_metrics: Some(server_cfg),
        network_latency: Some(network_latency_cfg),
        sidecars: Some(sidecars),
    };

    let benchmark_id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
    // The genai-perf-v1 envelope echoes the config; the runner treats it as an
    // opaque passthrough, so a projection of the native cfg (best-effort vs
    // Python's exclude_unset dump) keeps the aiperf-v1 exports emitting.
    let input_config = serde_json::to_value(&cfg).unwrap_or(serde_json::Value::Null);
    let mut export = crate::model::export::Export::build(
        &endpoint_type,
        true,
        &benchmark_id,
        input_config,
        serde_json::json!({}),
    );
    if let Some(url) = &inputs.otel_url {
        export.otel = Some(crate::model::export::OtelExport::build(
            url,
            &benchmark_id,
            &endpoint_type,
            &primary_model,
        ));
    }
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
        },
        kind,
    }
}

/// Parse repeatable `Name:value` header flags into a map (split on the first
/// colon; surrounding whitespace on the value trimmed, matching Python).
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

/// A default synthetic media dimension (`{value: 512}`) used when unset.
fn default_media_dim() -> Distribution {
    Distribution {
        value: Some(512.0),
        ..Default::default()
    }
}

/// Build the synthetic image spec when any `--image-*` flag is set.
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
        batch_size: flags.image_batch_size.unwrap_or(1),
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

/// Parse `--model-selection-strategy`.
fn parse_model_strategy(s: &str) -> anyhow::Result<ModelStrategy> {
    Ok(match s {
        "round_robin" => ModelStrategy::RoundRobin,
        "random" => ModelStrategy::Random,
        "weighted" => ModelStrategy::Weighted,
        other => anyhow::bail!("unknown --model-selection-strategy {other:?}"),
    })
}

/// Parse `--connection-reuse-strategy`.
fn parse_connection_reuse(s: &str) -> anyhow::Result<ConnectionReuse> {
    Ok(match s {
        "pooled" => ConnectionReuse::Pooled,
        "never" => ConnectionReuse::Never,
        "sticky-user-sessions" => ConnectionReuse::StickyUserSessions,
        other => anyhow::bail!("unknown --connection-reuse-strategy {other:?}"),
    })
}

/// Parse `--request-content-type` (MIME string) into the wire token.
fn parse_content_type(s: &str) -> anyhow::Result<RequestContentType> {
    Ok(match s {
        "application/json" => RequestContentType::ApplicationJson,
        "multipart/form-data" => RequestContentType::MultipartFormData,
        other => anyhow::bail!("unknown --request-content-type {other:?}"),
    })
}

/// Parse `--wait-for-model-mode`.
fn parse_wait_mode(s: &str) -> anyhow::Result<WaitForModelMode> {
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

/// Count the non-empty lines of a fixed-schedule input file (its entry count).
fn count_schedule_entries(path: &std::path::Path) -> anyhow::Result<u64> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read schedule {}: {e}", path.display()))?;
    Ok(text.lines().filter(|l| !l.trim().is_empty()).count() as u64)
}

/// Make a dataset path absolute (Python uses `path.absolute()`: cwd-join without
/// symlink resolution). Falls back to the input on cwd errors.
fn absolute_path(path: &std::path::Path) -> String {
    if path.is_absolute() {
        return path.to_string_lossy().into_owned();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(path).to_string_lossy().into_owned(),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

/// Normalize a base URL to include a scheme (Python prepends `http://` when the
/// user omits one, e.g. `127.0.0.1:8000` → `http://127.0.0.1:8000`).
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
        anyhow::bail!(
            "multi-run not yet supported: `{flag} {v}` describes a sweep; \
             run a single value or use the Python frontend"
        );
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
