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

use crate::config::model::DispatchMode;
use crate::config::model::artifacts::Artifacts;
use crate::config::model::dataset::{
    AudioSpec, Dataset, Distribution, ImageSpec, PrefixPrompts, PromptSelection, Prompts, Sampling,
    Synthetic, VideoSpec,
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
use crate::config::phase_validate::{apply_cli_loadgen_overlays, normalize_and_validate_phases};

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
    /// Emit the raw request/response JSONL (`artifacts.raw`).
    pub export_raw: bool,
    /// Emit per-request HTTP trace columns (`artifacts.trace`).
    pub export_trace: bool,
    /// Emit the per-request outputs JSON (`artifacts.export_outputs_json`).
    pub export_outputs_json: bool,
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
    if inputs.custom_dataset_type.as_deref() != Some("baseten_trace") {
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

pub fn resolve(mut inputs: Inputs) -> anyhow::Result<BenchmarkRun> {
    validate_baseten_only_trace_flags(&inputs)?;
    validate_baseten_extra_input_collisions(&inputs)?;
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
        Dataset::Public(crate::config::model::dataset::PublicDataset {
            cache_bust: None,
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
        if let Some(max) = crate::config::model::public_catalog::max_conversations(
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
        Dataset::Public(crate::config::model::dataset::PublicDataset {
            cache_bust: None,
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
        Dataset::File(crate::config::model::dataset::FileDataset {
            cache_bust: None,
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
                cache_bust: None,
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
                timing_mode: None,
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
            .map(|(rate, delay)| crate::config::model::phase::Cancellation { rate, delay });
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
    let mut network_latency_cfg = crate::config::model::telemetry::NetworkLatencyConfig::default();
    let network_latency_sidecar = if no_server_sidecars {
        None
    } else if let Some(mean_ms) = inputs.network_latency_mean {
        network_latency_cfg.enabled = true;
        network_latency_cfg.mean_ms = Some(mean_ms);
        Some(crate::config::model::telemetry::NetworkLatencySidecar::fixed(
            mean_ms,
        ))
    } else if let Some(ping) = inputs.network_latency_probe {
        network_latency_cfg.enabled = true;
        network_latency_cfg.ping_interval = ping;
        Some(crate::config::model::telemetry::NetworkLatencySidecar::probe(ping))
    } else {
        None
    };
    let sidecars = crate::config::model::telemetry::Sidecars {
        gpu_telemetry: gpu_enabled.then(|| {
            crate::config::model::telemetry::GpuTelemetrySidecar::default_dcgm(
                &inputs.gpu_telemetry_urls,
                inputs.gpu_telemetry_metrics_file.as_deref(),
            )
        }),
        server_metrics: server_enabled.then(|| {
            let mut all_urls = endpoint_urls.clone();
            all_urls.extend(inputs.server_metrics_urls.iter().cloned());
            let sc = crate::config::model::telemetry::ServerMetricsSidecar::from_endpoint_urls(&all_urls);
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
    let gpu_cfg = crate::config::model::telemetry::GpuTelemetryConfig {
        enabled: inputs.gpu_telemetry_enabled,
        urls: inputs.gpu_telemetry_urls.clone(),
        ..Default::default()
    };
    let server_cfg = crate::config::model::telemetry::ServerMetricsConfig {
        enabled: inputs.server_metrics_enabled,
        // Config preserves authored URLs; sidecar construction normalizes them.
        urls: inputs.server_metrics_urls.clone(),
        formats: inputs
            .server_metrics_formats
            .clone()
            .unwrap_or_else(|| crate::config::model::telemetry::ServerMetricsConfig::default().formats),
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
                .then(|| crate::config::model::metrics::SteadyState {
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
    crate::config::redact::redact_input_config(&mut input_config);
    let mut export = crate::config::model::export::Export::build(
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
        export.otel = Some(crate::config::model::export::OtelExport::build(
            url,
            &benchmark_id,
            &endpoint_type,
            &primary_model,
            inputs.otel_provider.as_deref(),
            &inputs.otel_resource_attributes,
        ));
    }
    export.mlflow = crate::config::model::export::MlflowExport::build(&inputs.mlflow, &benchmark_id);
    export.wandb = crate::config::model::export::WandbExport::build(&inputs.wandb, &benchmark_id);
    // Export formats use server-metrics config before export insertion.
    let sm_formats = cfg
        .server_metrics
        .as_ref()
        .map(|s| s.formats.clone())
        .unwrap_or_default();
    let sm_input_config = serde_json::to_value(&cfg).unwrap_or(serde_json::Value::Null);
    export.server_metrics = crate::config::model::export::ServerMetricsExport::build(
        &sm_formats,
        server_enabled,
        crate::config::model::export::AIPERF_V1_VERSION,
        &benchmark_id,
        sm_input_config,
    );
    export.parquet = crate::config::model::export::ParquetExport::build(&sm_formats, server_enabled);
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
    use crate::agentx::scenario::{apply_scenario_locks, get_scenario, RunLockInputs};

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
        crate::config::model::public_catalog::lookup(name).map(|meta| meta.format.clone())
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
        && let Some(spec) = crate::agentx::scenario::get_scenario(name)
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
