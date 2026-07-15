// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Pre-translation: parse `profile` flags (+ YAML, later) into the native
//! [`BenchmarkRun`].
//!
//! This is the reverse of Python's `rust_wire` projection: instead of lowering a
//! domain config into a wire dict, it builds the one native object (which *is*
//! the wire request) directly from the CLI surface. All the mapping — URL scheme
//! normalization, model→tokenizer defaulting, phase construction, unit handling,
//! and the resolved-defaults baseline — lives here.
//!
//! Defaults are the Python single-run synthetic defaults (proven byte-exact
//! against `tools/parity/golden/minimal_chat.request.json`). YAML merge and the
//! full flag surface are added incrementally; today this handles the CLI-only
//! synthetic path.

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
// `dataset/*`), proven against the minimal_chat golden.
const DEFAULT_TIMEOUT_SECONDS: f64 = 21600.0;
const DEFAULT_CONNECTION_LIMIT: u32 = 2500;
const DEFAULT_KEEPALIVE_TIMEOUT: f64 = 300.0;
const DEFAULT_WAIT_FOR_MODEL_INTERVAL: f64 = 5.0;
const DEFAULT_ISL_MEAN: f64 = 550.0;

/// Resolve `profile` flags into one native run, or a clear error.
///
/// Rejects multi-run (any comma-list sweep axis) since multi-run/orchestration
/// is deferred. YAML config files are not yet merged (returns an error if one is
/// supplied, so the caller can delegate to Python).
pub fn resolve(flags: &ProfileFlags) -> anyhow::Result<BenchmarkRun> {
    if flags.config_file.is_some() {
        anyhow::bail!("YAML config files are not yet supported by the native path");
    }
    reject_sweep("--concurrency", flags.concurrency.as_deref())?;
    reject_sweep("--request-count", flags.request_count.as_deref())?;
    reject_sweep("--request-rate", flags.request_rate.as_deref())?;

    anyhow::ensure!(
        !flags.model_names.is_empty(),
        "at least one --model is required"
    );
    anyhow::ensure!(!flags.urls.is_empty(), "at least one --url is required");
    let endpoint_type = flags
        .endpoint_type
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--endpoint-type is required"))?;

    let primary_model = flags.model_names[0].clone();
    let concurrency = parse_single::<u32>("--concurrency", flags.concurrency.as_deref())?;
    let request_count = parse_single::<u64>("--request-count", flags.request_count.as_deref())?;
    let request_rate = parse_single::<f64>("--request-rate", flags.request_rate.as_deref())?;

    let models = Models {
        strategy: ModelStrategy::RoundRobin,
        items: flags
            .model_names
            .iter()
            .map(|name| ModelItem {
                name: name.clone(),
                weight: None,
            })
            .collect(),
    };

    let endpoint = Endpoint {
        urls: flags.urls.iter().map(|u| normalize_url(u)).collect(),
        endpoint_type: EndpointType(endpoint_type),
        streaming: flags.streaming,
        use_legacy_max_tokens: false,
        use_server_token_count: false,
        timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
        connection_reuse: ConnectionReuse::Pooled,
        ssl_verify: true,
        connection_limit: DEFAULT_CONNECTION_LIMIT,
        keepalive_timeout: DEFAULT_KEEPALIVE_TIMEOUT,
        download_video_content: false,
        extra: serde_json::Map::new(),
        headers: std::collections::BTreeMap::new(),
        http2: false,
        wait_for_model_timeout: 0.0,
        wait_for_model_interval: DEFAULT_WAIT_FOR_MODEL_INTERVAL,
        wait_for_model_mode: WaitForModelMode::Inference,
        path: None,
        api_key: None,
        session_header: None,
        request_content_type: None,
        template: None,
        response_field: None,
    };

    let tokenizer = Tokenizer {
        // No --tokenizer flag yet: default to the primary model name (Python
        // uses "builtin" for fake model names — a later refinement).
        name: primary_model.clone(),
        revision: "main".to_string(),
        trust_remote_code: false,
        apply_chat_template: false,
    };

    let dataset = Dataset::Synthetic(Synthetic {
        prompts: Prompts {
            batch_size: 1,
            isl: Distribution {
                mean: Some(DEFAULT_ISL_MEAN),
                stddev: Some(0.0),
                ..Default::default()
            },
            osl: None,
            num_prefix_prompts: None,
            prefix_prompt_length: None,
        },
        sampling: Sampling("sequential".to_string()),
        turn_delay_ratio: 1.0,
        // The synthetic corpus is sized to the request bound (one entry per
        // request); Python projects `entries == request_count` on this path.
        entries: request_count.map(|n| n as u32),
        num_conversations: None,
        turn_delay_ms: None,
    });

    // One profiling phase. A request rate selects a Poisson arrival phase;
    // otherwise a fixed-concurrency phase (default concurrency 1).
    let kind = if let Some(rate) = request_rate {
        PhaseKind::Poisson { rate, concurrency }
    } else {
        PhaseKind::Concurrency {
            concurrency: concurrency.unwrap_or(1),
        }
    };
    let phase = Phase {
        common: PhaseCommon {
            name: "profiling".to_string(),
            exclude_from_results: false,
            seamless: false,
            requests: request_count,
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
        kind,
    };

    let cfg = BenchmarkConfig {
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
        phases: Some(vec![phase]),
    };

    let artifact_dir = flags
        .artifact_dir
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("artifacts"));

    Ok(BenchmarkRun {
        benchmark_id: uuid::Uuid::new_v4().simple().to_string()[..12].to_string(),
        artifact_dir,
        cfg,
        cli_command: None,
        label: String::new(),
        random_seed: None,
        sweep_id: None,
        trial: 0,
        variation: None,
        resolved: Resolved::default(),
        variables: serde_json::Map::new(),
    })
}

/// Normalize a base URL to include a scheme (Python prepends `http://` when the
/// user omits one, e.g. `127.0.0.1:8000` → `http://127.0.0.1:8000`).
fn normalize_url(url: &str) -> String {
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
