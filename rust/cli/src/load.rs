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
use crate::model::dataset::{Dataset, Distribution, Prompts, Sampling, Synthetic};
use crate::model::endpoint::{ConnectionReuse, Endpoint, EndpointType, WaitForModelMode};
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
}

/// Normalized inputs both surfaces (flags / YAML) resolve to before building.
pub(crate) struct Inputs {
    pub model_names: Vec<String>,
    pub urls: Vec<String>,
    pub endpoint_type: String,
    pub streaming: bool,
    pub api_key: Option<String>,
    pub headers: std::collections::BTreeMap<String, String>,
    pub tokenizer_name: Option<String>,
    pub tokenizer_revision: Option<String>,
    pub tokenizer_trust: bool,
    pub isl: Distribution,
    pub osl: Option<Distribution>,
    pub batch_size: u32,
    pub sampling: String,
    pub entries: u32,
    /// Profiling-phase session bound (from `num_conversations`).
    pub sessions: Option<u64>,
    pub concurrency: Option<u32>,
    pub request_rate: Option<f64>,
    pub request_count: Option<u64>,
    pub benchmark_duration: Option<f64>,
    pub grace_period: Option<f64>,
    pub warmup: Option<Warmup>,
    pub random_seed: Option<u64>,
    /// File-backed dataset path (mutually exclusive with the synthetic path).
    pub input_file: Option<PathBuf>,
    /// File dataset format id (`--custom-dataset-type`).
    pub custom_dataset_type: Option<String>,
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
    reject_sweep("--isl", flags.isl.as_deref())?;
    reject_sweep("--osl", flags.osl.as_deref())?;
    let isl_mean = parse_single::<f64>("--isl", flags.isl.as_deref())?;
    let osl_mean = parse_single::<f64>("--osl", flags.osl.as_deref())?;

    let warmup = if flags.warmup_request_count.is_none()
        && flags.warmup_concurrency.is_none()
        && flags.warmup_request_rate.is_none()
    {
        None
    } else {
        Some(Warmup {
            concurrency: flags.warmup_concurrency,
            rate: flags.warmup_request_rate,
            requests: flags.warmup_request_count,
        })
    };

    let inputs = Inputs {
        model_names: flags.model_names.clone(),
        urls: flags.urls.clone(),
        endpoint_type,
        streaming: flags.streaming,
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
        batch_size: 1,
        sampling: "sequential".to_string(),
        entries: num_conversations
            .or(request_count.map(|n| n as u32))
            .unwrap_or(DEFAULT_ENTRIES),
        sessions: num_conversations.map(u64::from),
        concurrency,
        request_rate,
        request_count,
        benchmark_duration,
        grace_period: flags.benchmark_grace_period,
        warmup,
        random_seed: flags.random_seed,
        input_file: flags.input_file.clone(),
        custom_dataset_type: flags.custom_dataset_type.clone(),
        artifact_dir: flags
            .artifact_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("artifacts")),
    };
    Ok(build(inputs))
}

/// Build the one native run from normalized inputs. This is the single place the
/// wire defaults live; both the flag and YAML surfaces funnel through here.
pub(crate) fn build(inputs: Inputs) -> BenchmarkRun {
    let primary_model = inputs.model_names[0].clone();

    let models = Models {
        strategy: ModelStrategy::RoundRobin,
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
        use_legacy_max_tokens: false,
        use_server_token_count: false,
        timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
        connection_reuse: ConnectionReuse::Pooled,
        ssl_verify: true,
        connection_limit: DEFAULT_CONNECTION_LIMIT,
        keepalive_timeout: DEFAULT_KEEPALIVE_TIMEOUT,
        download_video_content: false,
        extra: serde_json::Map::new(),
        headers: inputs.headers,
        http2: false,
        wait_for_model_timeout: 0.0,
        wait_for_model_interval: DEFAULT_WAIT_FOR_MODEL_INTERVAL,
        wait_for_model_mode: WaitForModelMode::Inference,
        path: None,
        api_key: inputs.api_key,
        session_header: None,
        request_content_type: None,
        template: None,
        response_field: None,
    };

    let tokenizer = Tokenizer {
        name: inputs.tokenizer_name.unwrap_or(primary_model),
        revision: inputs
            .tokenizer_revision
            .unwrap_or_else(|| "main".to_string()),
        trust_remote_code: inputs.tokenizer_trust,
        apply_chat_template: false,
    };

    let dataset = match &inputs.input_file {
        Some(path) => Dataset::File(crate::model::dataset::FileDataset {
            format: inputs
                .custom_dataset_type
                .clone()
                .unwrap_or_else(|| "single_turn".to_string()),
            sampling: Sampling(inputs.sampling.clone()),
            options: serde_json::Map::new(),
            path: Some(absolute_path(path)),
            entries: Some(inputs.entries),
            random_seed: inputs.random_seed,
            osl: inputs.osl.clone(),
        }),
        None => Dataset::Synthetic(Synthetic {
            prompts: Prompts {
                batch_size: inputs.batch_size,
                isl: inputs.isl.clone(),
                osl: inputs.osl.clone(),
                num_prefix_prompts: None,
                prefix_prompt_length: None,
            },
            sampling: Sampling(inputs.sampling.clone()),
            turn_delay_ratio: 1.0,
            entries: Some(inputs.entries),
            num_conversations: None,
            turn_delay_ms: None,
        }),
    };

    let profiling = build_phase(
        "profiling",
        false,
        inputs.concurrency.unwrap_or(1),
        inputs.request_rate,
        inputs.concurrency,
        inputs.request_count,
        inputs.sessions,
        inputs.benchmark_duration,
        inputs.grace_period,
    );
    let mut phases = Vec::new();
    if let Some(warmup) = inputs.warmup {
        let concurrency = warmup.concurrency.or(inputs.concurrency);
        phases.push(build_phase(
            "warmup",
            true,
            concurrency.unwrap_or(1),
            warmup.rate,
            concurrency,
            warmup.requests,
            None,
            None,
            None,
        ));
    }
    phases.push(profiling);

    let endpoint_type = endpoint.endpoint_type.0.clone();
    let endpoint_urls = endpoint.urls.clone();
    // GPU telemetry + server-metrics scraping are enabled by default; lower them
    // into the sidecars block (server-metrics scrapes each endpoint's /metrics,
    // GPU telemetry scrapes the default DCGM endpoints).
    let sidecars = crate::model::telemetry::Sidecars {
        gpu_telemetry: Some(crate::model::telemetry::GpuTelemetrySidecar::default_dcgm()),
        server_metrics: Some(
            crate::model::telemetry::ServerMetricsSidecar::from_endpoint_urls(&endpoint_urls),
        ),
    };
    let mut cfg = BenchmarkConfig {
        models: Some(models),
        endpoint: Some(endpoint),
        tokenizer: Some(tokenizer),
        transport: Some(Transport::Http),
        runtime: Some(Runtime::default()),
        metrics: Some(Metrics::default()),
        artifacts: Some(Artifacts {
            trace: false,
            inputs_path: "inputs.json".to_string(),
            records_path: Some("profile_export.jsonl".to_string()),
            ..Default::default()
        }),
        datasets: Some(vec![dataset]),
        phases: Some(phases),
        export: None,
        gpu_telemetry: Some(crate::model::telemetry::GpuTelemetryConfig::default()),
        server_metrics: Some(crate::model::telemetry::ServerMetricsConfig::default()),
        network_latency: Some(crate::model::telemetry::NetworkLatencyConfig::default()),
        sidecars: Some(sidecars),
    };

    let benchmark_id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
    // The genai-perf-v1 envelope echoes the config; the runner treats it as an
    // opaque passthrough, so a projection of the native cfg (best-effort vs
    // Python's exclude_unset dump) keeps the aiperf-v1 exports emitting.
    let input_config = serde_json::to_value(&cfg).unwrap_or(serde_json::Value::Null);
    cfg.export = Some(crate::model::export::Export::build(
        &endpoint_type,
        true,
        &benchmark_id,
        input_config,
        serde_json::json!({}),
    ));

    BenchmarkRun {
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
    }
}

/// Build one phase from resolved axes. A request rate selects a Poisson arrival
/// phase (with optional concurrency cap); otherwise a fixed-concurrency phase.
#[allow(clippy::too_many_arguments)]
fn build_phase(
    name: &str,
    exclude_from_results: bool,
    default_concurrency: u32,
    rate: Option<f64>,
    concurrency: Option<u32>,
    requests: Option<u64>,
    sessions: Option<u64>,
    duration: Option<f64>,
    grace_period: Option<f64>,
) -> Phase {
    let kind = if let Some(rate) = rate {
        PhaseKind::Poisson { rate, concurrency }
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
