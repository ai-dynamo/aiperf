// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! YAML Config v2 parsing and normalization.
//!
//! Config v2 accepts shorthand forms (`model:` → `models.items[0]`, `dataset:`
//! → `datasets[0]`, a flat `phases:` → one phase) and both snake_case and
//! camelCase keys via `serde(alias)`. Unknown keys are ignored.

use std::path::PathBuf;

use serde::Deserialize;

use crate::load::{self, Inputs, Warmup, default_isl};
use crate::model::dataset::Distribution;
use crate::model::endpoint::{ResetKvCacheConfig, ServerProfilerConfig};
use crate::model::transport::{DynosimConfig, Transport};

/// Parse a YAML config file into one native run.
pub fn resolve(
    path: &std::path::Path,
    artifact_dir: Option<PathBuf>,
) -> anyhow::Result<crate::model::BenchmarkRun> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read config {}: {e}", path.display()))?;
    resolve_str(&text, artifact_dir)
        .map_err(|e| anyhow::anyhow!("in config {}: {e}", path.display()))
}

/// Parse YAML config text into one native run (the file-independent core).
pub(crate) fn resolve_str(
    text: &str,
    artifact_dir: Option<PathBuf>,
) -> anyhow::Result<crate::model::BenchmarkRun> {
    let raw: serde_json::Value =
        serde_yaml::from_str(text).map_err(|e| anyhow::anyhow!("failed to parse config: {e}"))?;
    let expanded = crate::expand::expand_config(raw)?;
    resolve_expanded_value(expanded, artifact_dir, None)
}

/// Resolve an already `${ENV}`+Jinja-expanded config value into one run. Shared
/// by the single-run path and the YAML `sweep:` path (which env-substitutes once,
/// overrides per variation, then Jinja-renders each variation before this call).
pub(crate) fn resolve_expanded_value(
    expanded: serde_json::Value,
    artifact_dir: Option<PathBuf>,
    overrides: Option<&crate::flags::ProfileFlags>,
) -> anyhow::Result<crate::model::BenchmarkRun> {
    load::build(resolve_expanded_inputs(expanded, artifact_dir, overrides)?)
}

/// Normalize an already-expanded config value into authoring [`Inputs`] without
/// resolving. The single-run path serializes these onto the `--execute` wire so the
/// runtime resolves; [`resolve_expanded_value`] additionally builds the run for the
/// YAML `sweep:` path, which resolves CLI-side.
pub(crate) fn resolve_expanded_inputs(
    expanded: serde_json::Value,
    artifact_dir: Option<PathBuf>,
    overrides: Option<&crate::flags::ProfileFlags>,
) -> anyhow::Result<Inputs> {
    let mut file: ConfigFile = serde_json::from_value(expanded)
        .map_err(|e| anyhow::anyhow!("failed to parse config: {e}"))?;
    let random_seed = file.random_seed;
    // A nested runtime block takes precedence over the top-level block.
    if file.benchmark.runtime.is_none() {
        file.benchmark.runtime = file.runtime.take();
    }
    let mut inputs = file.benchmark.into_inputs(artifact_dir, random_seed)?;
    apply_cli_overrides(&mut inputs, overrides)?;
    Ok(inputs)
}

/// Overlay an explicitly-set `Option<bool>` flag onto a config-derived `bool`.
///
/// A `None` flag (unset) is a no-op, so the config-authored value (or its
/// effective default) is preserved byte-for-byte; only an explicitly authored
/// `--flag`/`--flag=false` overrides. This is the single mechanism the operational
/// bool toggles below drive through instead of a per-flag `unwrap_or`/`if` line.
fn overlay_bool(slot: &mut bool, flag: Option<bool>) {
    if let Some(v) = flag {
        *slot = v;
    }
}

/// Apply explicitly authored operational CLI flags over a config-derived run;
/// model, dataset, and phase content remains config-owned. Operational endpoint
/// and dataset bool toggles (`--streaming`, `--use-server-token-count`, …) overlay
/// only when explicitly set, mirroring the flag-only path's `Inputs` mapping.
fn apply_cli_overrides(
    inputs: &mut Inputs,
    overrides: Option<&crate::flags::ProfileFlags>,
) -> anyhow::Result<()> {
    let Some(flags) = overrides else {
        return Ok(());
    };
    // Operational bool toggles that map directly to a single `Inputs` bool, driven
    // through one mechanism. Each overlays only when explicitly set (`Some`); an
    // unset flag leaves the config-authored value (or default) unchanged. Inverse
    // `--no-*`-paired toggles (open_loop_replay, force_min_tokens, auto_plot …) and
    // compound toggles (steady_state windowing, dataset-analysis knobs) resolve
    // bespoke below and are intentionally not folded here.
    overlay_bool(&mut inputs.streaming, flags.streaming);
    overlay_bool(
        &mut inputs.use_legacy_max_tokens,
        flags.use_legacy_max_tokens,
    );
    overlay_bool(
        &mut inputs.use_server_token_count,
        flags.use_server_token_count,
    );
    overlay_bool(
        &mut inputs.download_video_content,
        flags.download_video_content,
    );
    overlay_bool(&mut inputs.apply_chat_template, flags.apply_chat_template);
    overlay_bool(&mut inputs.proxy_from_env, flags.proxy_from_env);
    overlay_bool(&mut inputs.prefetch_media_urls, flags.prefetch_media_urls);
    overlay_bool(&mut inputs.uuid_and_strip, flags.uuid_and_strip);
    overlay_bool(&mut inputs.omit_kv_hints, flags.omit_kv_hints);
    overlay_bool(&mut inputs.open_loop_strict, flags.open_loop_strict);
    overlay_bool(&mut inputs.unsafe_override, flags.unsafe_override);
    overlay_bool(&mut inputs.export_outputs_json, flags.export_outputs_json);
    overlay_bool(&mut inputs.show_trace_timing, flags.show_trace_timing);
    overlay_bool(&mut inputs.use_think_time_only, flags.use_think_time_only);
    overlay_bool(&mut inputs.burst_phase_starts, flags.burst_phase_starts);
    if flags.show_trace_timing.unwrap_or(false) {
        inputs.export_trace = true;
    }
    if let Some(prefix) = flags.profile_export_prefix.clone() {
        inputs.profile_export_prefix = Some(prefix);
    }
    if let Some(v) = flags.max_context_length {
        inputs.max_context_length = Some(v);
    }
    if let Some(v) = flags.trace_idle_gap_cap_seconds {
        inputs.trace_idle_gap_cap_seconds = Some(v);
    }
    if let Some(v) = flags.system_idle_gap_cap_seconds {
        inputs.system_idle_gap_cap_seconds = Some(v);
    }
    if let Some(v) = flags.cache_bust.clone().filter(|t| t != "none") {
        inputs.cache_bust = Some(v);
    }
    if flags.allow_dataset_wrap.unwrap_or(false) {
        inputs.allow_dataset_wrap = Some(true);
    } else if flags.no_allow_dataset_wrap.unwrap_or(false) {
        inputs.allow_dataset_wrap = Some(false);
    }
    if let Some(repo) = flags.hf_weka_dataset.clone() {
        if let Some(name) = inputs.public_dataset.as_deref()
            && name != "weka_hf"
        {
            anyhow::bail!(
                "--hf-weka-dataset cannot be combined with --public-dataset {name}; omit --public-dataset or set it to weka_hf"
            );
        }
        inputs.public_dataset = Some("weka_hf".to_string());
        inputs.hf_weka_dataset = Some(repo);
    }
    if let Some(v) = flags.trace_session_sample_ratio {
        inputs.trace_session_sample_ratio = Some(v);
    }
    if let Some(v) = flags.agentic_warmup_grace_period {
        inputs.agentic_warmup_grace_period = Some(v);
    }
    if let Some(v) = flags.failed_request_threshold {
        inputs.failed_request_threshold = Some(v);
    }
    if flags.use_think_time_only.unwrap_or(false) && flags.ignore_trace_delays.unwrap_or(false) {
        anyhow::bail!("--use-think-time-only and --ignore-trace-delays are mutually exclusive");
    }
    if let Some(name) = flags.tokenizer.clone() {
        inputs.tokenizer_name = Some(name);
    }
    if let Some(server_url) = flags.server_tokenizer_url.clone() {
        inputs.server_tokenizer_url = Some(server_url);
    }
    if let Some(corpus) = flags.prompt_corpus.clone() {
        inputs.prompt_corpus = Some(corpus);
    }
    load::overlay_reset_kv_cache_config(
        &mut inputs.reset_kv_cache,
        load::reset_kv_cache_from_flags(flags),
    );
    load::overlay_server_profiler_config(
        &mut inputs.server_profiler,
        load::server_profiler_from_flags(flags),
    );
    if let Some(level) = flags.export_level.as_deref() {
        let (records_formats, export_raw) = load::export_level_formats(Some(level))?;
        inputs.records_formats = records_formats;
        inputs.export_raw = export_raw;
    }
    if let Some(cells) = flags.cells {
        inputs.runtime_cells = cells;
    }
    // Same precedence as `--cells` over `runtime.cells`: an explicit `--dispatch`
    // wins over an authored `runtime.dispatch`.
    if flags.dispatch.is_some() {
        inputs.runtime_dispatch = Some(flags.dispatch_mode()?);
    }
    // An explicit `--hop-routing` wins over an authored `runtime.hop_routing`.
    if flags.hop_routing.is_some() {
        inputs.runtime_hop_routing = flags.hop_routing()?;
    }
    // CLI random seed governs both run and dataset sampling.
    if let Some(seed) = flags.random_seed {
        inputs.random_seed = Some(seed);
        inputs.dataset_random_seed = Some(seed);
    }
    if !flags.server_metrics_formats.is_empty() {
        inputs.server_metrics_formats = Some(flags.server_metrics_formats.clone());
    }
    // Steady-state windowing: `--steady-state` (+ optional `--steady-state-fraction`)
    // layers over a config-authored run.
    overlay_bool(&mut inputs.steady_state, flags.steady_state);
    if let Some(fraction) = flags.steady_state_fraction {
        inputs.steady_state_fraction = Some(fraction);
    }
    overlay_bool(&mut inputs.steady_state_hybrid, flags.steady_state_hybrid);
    // Explicit CLI loadgen axes overlay onto a unique profiling phase when the
    // config authored a multi-phase workflow (`phases_override`).
    if let Some(v) = load::parse_single::<u32>("--concurrency", flags.concurrency.as_deref())? {
        inputs.concurrency = Some(v);
    }
    if let Some(v) = load::parse_single::<u64>("--request-count", flags.request_count.as_deref())? {
        inputs.request_count = Some(v);
    }
    if let Some(v) = load::parse_single::<f64>("--request-rate", flags.request_rate.as_deref())? {
        inputs.request_rate = Some(v);
    }
    if let Some(v) =
        load::parse_single::<f64>("--benchmark-duration", flags.benchmark_duration.as_deref())?
    {
        inputs.benchmark_duration = Some(v);
    }
    if let Some(v) = flags.benchmark_grace_period {
        inputs.grace_period = Some(v);
    }
    if let Some(v) = flags.prefill_concurrency {
        inputs.prefill_concurrency = Some(v);
    }
    if let Some(v) =
        load::parse_single::<u32>("--num-conversations", flags.num_conversations.as_deref())?
    {
        inputs.sessions = Some(u64::from(v));
    }
    if flags.request_rate_mode.is_some() {
        inputs.rate_mode = flags.request_rate_mode.clone();
    }
    if let Some(series) = load::resolve_request_rate_series(
        flags.request_rate_series.as_ref(),
        inputs.request_rate,
        inputs.user_centric.is_some(),
    )? {
        inputs.request_rate_series = Some(series);
    }
    // Dry-run dataset-analysis toggles: `--no-dataset-analysis` suppresses the
    // family; the `--kv-*` / `--dataset-analysis-per-conversation` knobs layer
    // over the config-derived defaults when the analysis is active.
    if flags.no_dataset_analysis.unwrap_or(false) {
        inputs.dataset_analysis = None;
    } else if let Some(analysis) = inputs.dataset_analysis.as_mut() {
        // `--kv-block-size` carries its default (16); only override when authored.
        if flags.kv_block_size != 16 {
            analysis.block_size = flags.kv_block_size;
        }
        if let Some(cache_blocks) = flags.kv_cache_blocks {
            analysis.cache_blocks = Some(cache_blocks);
        }
        if flags.dataset_analysis_per_conversation.unwrap_or(false) {
            analysis.per_conversation = true;
        }
    }
    Ok(())
}

/// Parse a config and substitute `${ENV}` before sweep and Jinja expansion.
pub fn read_env_substituted(path: &std::path::Path) -> anyhow::Result<serde_json::Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read config {}: {e}", path.display()))?;
    let raw: serde_json::Value = serde_yaml::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse config {}: {e}", path.display()))?;
    crate::expand::substitute_env(raw)
}

/// A string or a list of strings (Config shorthand for single-vs-many).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

impl StringOrVec {
    fn into_vec(self) -> Vec<String> {
        match self {
            StringOrVec::One(s) => vec![s],
            StringOrVec::Many(v) => v,
        }
    }
}

/// A scalar or a parametric distribution (Config shorthand, e.g. `isl: 512`).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum NumOrDist {
    Num(f64),
    Dist(DistFields),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EnabledOrConfig<T> {
    Enabled(bool),
    Config(T),
}

#[derive(Debug, Deserialize)]
struct DistFields {
    /// Scalar fixed value (`{value: N}` — the object form of the `isl: N`
    /// shorthand, e.g. what the Kubernetes operator projects for a fixed
    /// synthetic ISL/OSL). Selects a `Fixed` distribution.
    value: Option<f64>,
    mean: Option<f64>,
    stddev: Option<f64>,
    /// Median selects a log-normal distribution (paired with `mean`).
    median: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

/// Deserialize seconds from a number, `30s`, `5m`, `2h`, or `inf`.
fn de_duration_opt<'de, D>(d: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum DurOrNum {
        Num(f64),
        Str(String),
    }
    match Option::<DurOrNum>::deserialize(d)? {
        None => Ok(None),
        Some(DurOrNum::Num(n)) => Ok(Some(n)),
        Some(DurOrNum::Str(s)) => load::parse_duration_secs(&s)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

#[derive(Debug, Deserialize)]
struct ConfigFile {
    benchmark: Benchmark,
    /// Top-level deterministic run seed (`randomSeed`).
    #[serde(default, alias = "randomSeed")]
    random_seed: Option<u64>,
    /// Top-level worker/cell runtime policy (`runtime.cells`).
    #[serde(default)]
    runtime: Option<RuntimeSection>,
}

#[derive(Debug, Deserialize)]
struct Benchmark {
    /// `model:` shorthand (single string or list).
    model: Option<StringOrVec>,
    /// Expanded `models:` block.
    models: Option<ModelsSection>,
    endpoint: EndpointSection,
    /// Orthogonal transport selection (`http` default; `grpc`/`dynosim_*`).
    transport: Option<TransportSection>,
    /// `dataset:` shorthand (single entry).
    dataset: Option<DatasetSection>,
    /// Expanded `datasets:` list (first entry used on the single-run path).
    datasets: Option<Vec<DatasetSection>>,
    tokenizer: Option<TokenizerSection>,
    /// Advanced multi-phase list (mutually exclusive with `warmup`/`profiling`).
    phases: Option<Phases>,
    /// Simple-config leading warmup phase (paired with `profiling`).
    warmup: Option<PhaseSection>,
    /// Simple-config profiling phase (the `warmup`/`profiling` form).
    profiling: Option<PhaseSection>,
    /// Output artifacts block (`dir` is the run's artifact target).
    artifacts: Option<ArtifactsSection>,
    /// Goodput SLO thresholds (`benchmark.slos`: metric -> ms).
    slos: Option<std::collections::BTreeMap<String, f64>>,
    /// GPU telemetry policy.
    #[serde(default, alias = "gpuTelemetry")]
    gpu_telemetry: Option<GpuTelemetrySection>,
    /// Server-metrics scraping policy.
    #[serde(default, alias = "serverMetrics")]
    server_metrics: Option<ServerMetricsSection>,
    /// Network-latency calibration policy.
    #[serde(default, alias = "networkLatency")]
    network_latency: Option<NetworkLatencySection>,
    /// OTLP export sink.
    otel: Option<OtelSection>,
    /// MLflow export sink.
    mlflow: Option<MlflowSection>,
    /// Weights & Biases export sink.
    wandb: Option<WandbSection>,
    /// Worker/cell runtime policy.
    runtime: Option<RuntimeSection>,
    /// Named submission scenario (`cfg.scenario`).
    scenario: Option<String>,
    /// Recorded-graph trajectory-start window lower ratio.
    #[serde(default, alias = "trajectoryStartMinRatio")]
    trajectory_start_min_ratio: Option<f64>,
    /// Recorded-graph trajectory-start window upper ratio.
    #[serde(default, alias = "trajectoryStartMaxRatio")]
    trajectory_start_max_ratio: Option<f64>,
    /// Relax cross-field validation (`cfg.unsafe_override`).
    #[serde(default, alias = "unsafeOverride")]
    unsafe_override: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ArtifactsSection {
    /// Run artifact directory (the `--artifact-dir` flag overrides it).
    dir: Option<String>,
    /// Timeslice window, seconds (wire `metrics.slice_duration_seconds`).
    #[serde(default, alias = "sliceDuration", deserialize_with = "de_duration_opt")]
    slice_duration: Option<f64>,
    /// Per-record export formats (`[jsonl,csv,parquet]`) or `false` to disable.
    records: Option<RecordsFormats>,
    /// Emit the raw request/response JSONL.
    #[serde(default)]
    raw: bool,
    /// Emit per-request HTTP trace columns.
    #[serde(default)]
    trace: bool,
    /// Emit the per-request outputs JSON.
    #[serde(default, alias = "exportOutputsJson")]
    export_outputs_json: bool,
    /// Show per-request trace timing in the console.
    #[serde(default, alias = "showTraceTiming")]
    show_trace_timing: bool,
    /// Base filename stem for exported artifacts (`artifacts.prefix`).
    #[serde(default)]
    prefix: Option<String>,
}

/// `artifacts.records`: a format list, or `false` to disable per-record export.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RecordsFormats {
    List(Vec<String>),
    /// `records: false` disables per-record export; the bool is only a
    /// deserialization discriminant, never read.
    Disabled(#[allow(dead_code)] bool),
}

#[derive(Debug, Deserialize)]
struct GpuTelemetrySection {
    enabled: Option<bool>,
    urls: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct ServerMetricsSection {
    enabled: Option<bool>,
    urls: Option<Vec<String>>,
    formats: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct NetworkLatencySection {
    #[serde(default)]
    enabled: bool,
    #[serde(default, alias = "meanMs")]
    mean_ms: Option<f64>,
    #[serde(default, alias = "pingInterval")]
    ping_interval: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OtelSection {
    #[serde(alias = "metricsUrl")]
    metrics_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MlflowSection {
    #[serde(default, alias = "trackingUri")]
    tracking_uri: Option<String>,
    experiment: Option<String>,
    #[serde(default, alias = "runName")]
    run_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WandbSection {
    project: Option<String>,
    entity: Option<String>,
    #[serde(default, alias = "runName")]
    run_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RuntimeSection {
    workers: Option<u32>,
    #[serde(default, alias = "workersMin")]
    workers_min: Option<u32>,
    cells: Option<u32>,
    /// Admission strategy for `workers>1` scheduled execution (`sharded`/`global`/
    /// `global-hop`). Absent selects [`DispatchMode::default`] (`Global`); reuses
    /// `DispatchMode`'s own `Deserialize` impl so YAML and `--dispatch` validate
    /// identically instead of duplicating the accepted-value list here.
    #[serde(default)]
    dispatch: Option<aiperf_runtime::engine::protocol::DispatchMode>,
    /// Worker-assignment policy for `dispatch == global-hop` with `workers > 1`
    /// (`round-robin`/`sticky`/`least-loaded`). Absent lets resolution derive it
    /// from the connection-reuse strategy; reuses `HopRouting`'s own
    /// `Deserialize` impl so YAML and `--hop-routing` validate identically.
    #[serde(default, alias = "hopRouting")]
    hop_routing: Option<aiperf_runtime::engine::protocol::HopRouting>,
}

/// A full models mapping or shorthand sequence of model names or item maps.
#[derive(Debug)]
struct ModelsSection {
    items: Vec<ModelItem>,
    /// Model-selection strategy (`round_robin`/`random`/`weighted`).
    strategy: Option<String>,
}

impl<'de> Deserialize<'de> for ModelsSection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = ModelsSection;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a `models` mapping or a sequence of model names")
            }
            fn visit_seq<A>(self, seq: A) -> Result<ModelsSection, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let items = Vec::<ModelItem>::deserialize(
                    serde::de::value::SeqAccessDeserializer::new(seq),
                )?;
                Ok(ModelsSection {
                    items,
                    strategy: None,
                })
            }
            fn visit_map<A>(self, map: A) -> Result<ModelsSection, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                #[derive(Deserialize)]
                struct Full {
                    items: Vec<ModelItem>,
                    #[serde(default)]
                    strategy: Option<String>,
                }
                let full = Full::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
                Ok(ModelsSection {
                    items: full.items,
                    strategy: full.strategy,
                })
            }
        }
        deserializer.deserialize_any(V)
    }
}

/// One selectable model. Accepts a bare model-name string (`gpt-4`) or a
/// `{name: gpt-4, ...}` mapping.
#[derive(Debug)]
struct ModelItem {
    name: String,
}

impl<'de> Deserialize<'de> for ModelItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = ModelItem;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a model-name string or a `{name, ...}` mapping")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<ModelItem, E> {
                Ok(ModelItem {
                    name: v.to_string(),
                })
            }
            fn visit_map<A>(self, map: A) -> Result<ModelItem, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                #[derive(Deserialize)]
                struct Full {
                    name: String,
                }
                let full = Full::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
                Ok(ModelItem { name: full.name })
            }
        }
        deserializer.deserialize_any(V)
    }
}

#[derive(Debug, Deserialize)]
struct TransportSection {
    #[serde(rename = "type")]
    transport_type: String,
    /// DynoSim knobs sit flat on the transport object (like Config's
    /// `DynosimOfflineTransport`); captured for the `dynosim_*` types and
    /// ignored (all-`None`) for `http`/`grpc`.
    #[serde(flatten)]
    dynosim: DynosimConfig,
    /// `dry_run` analytic-latency knobs sit flat on the transport object;
    /// captured for the `dry_run` type and ignored (all-`None`) otherwise.
    #[serde(flatten)]
    dry_run: crate::model::transport::DryRunConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResetKvCacheYaml {
    #[serde(default, alias = "timeoutSeconds")]
    timeout_seconds: Option<f64>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServerProfilerYaml {
    #[serde(default, alias = "timeoutSeconds")]
    timeout_seconds: Option<f64>,
    #[serde(default, alias = "startPath")]
    start_path: Option<String>,
    #[serde(default, alias = "stopPath")]
    stop_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EndpointSection {
    #[serde(rename = "type")]
    endpoint_type: Option<String>,
    /// Endpoint URL(s); `url:` (single/list) or plural `urls:`. Optional:
    /// DynoSim endpoints carry no URL (the sentinel is injected).
    #[serde(alias = "urls")]
    url: Option<StringOrVec>,
    #[serde(default)]
    streaming: bool,
    #[serde(default, alias = "apiKey")]
    api_key: Option<String>,
    /// Request timeout, seconds (wire `timeout_seconds`).
    timeout: Option<f64>,
    #[serde(default, alias = "connectionReuse")]
    connection_reuse: Option<String>,
    #[serde(
        default,
        rename = "use_legacy_max_tokens",
        alias = "useLegacyMaxTokens"
    )]
    use_legacy_max_tokens: Option<bool>,
    #[serde(default, alias = "useServerTokenCount")]
    use_server_token_count: Option<bool>,
    #[serde(default, alias = "downloadVideoContent")]
    download_video_content: Option<bool>,
    #[serde(default)]
    headers: Option<std::collections::BTreeMap<String, String>>,
    #[serde(default)]
    extra: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default, alias = "requestContentType")]
    request_content_type: Option<String>,
    #[serde(default, alias = "sessionHeader")]
    session_header: Option<String>,
    /// Explicit forward-proxy URL for benchmark traffic.
    #[serde(default)]
    proxy: Option<String>,
    /// Honor the ambient proxy environment for benchmark traffic.
    #[serde(default, alias = "proxyFromEnv")]
    proxy_from_env: Option<bool>,
    /// Custom request path appended to the endpoint URL (`endpoint.path`).
    path: Option<String>,
    #[serde(default, alias = "resetKvCache")]
    reset_kv_cache: Option<EnabledOrConfig<ResetKvCacheYaml>>,
    #[serde(default, alias = "serverProfiler")]
    server_profiler: Option<EnabledOrConfig<ServerProfilerYaml>>,
    #[serde(default, alias = "waitForModelTimeout")]
    wait_for_model_timeout: Option<f64>,
    #[serde(default, alias = "waitForModelInterval")]
    wait_for_model_interval: Option<f64>,
    #[serde(default, alias = "waitForModelMode")]
    wait_for_model_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DatasetSection {
    /// `synthetic` (default), `file`, or `public`; drives the loader branch.
    #[serde(rename = "type")]
    dataset_type: Option<String>,
    /// Public dataset catalog name (`dataset.dataset`, when `type: public`).
    #[serde(rename = "dataset")]
    public_name: Option<String>,
    /// HuggingFace subset override for the public dataset.
    #[serde(default, alias = "hfSubset")]
    hf_subset: Option<String>,
    /// Arbitrary Hugging Face dataset repository ID (bypasses the catalog).
    #[serde(default, alias = "hfDataset")]
    hf_dataset: Option<String>,
    /// Hugging Face dataset split (auto-resolved if omitted).
    #[serde(default, alias = "hfSplit")]
    hf_split: Option<String>,
    /// Hugging Face dataset git revision.
    #[serde(default, alias = "hfRevision")]
    hf_revision: Option<String>,
    /// Forced prompt column for the auto-detecting `hf` format.
    #[serde(default, alias = "hfTextColumn")]
    hf_text_column: Option<String>,
    /// Forced completion/output column for the auto-detecting `hf` format.
    #[serde(default, alias = "hfOutputColumn")]
    hf_output_column: Option<String>,
    /// Fixed output length for `hf_dataset`.
    #[serde(default, alias = "hfOutputLen")]
    hf_output_len: Option<u32>,
    /// File-dataset path (trace/replay).
    path: Option<String>,
    /// Inline file-dataset records authored in the config (instead of `path:`).
    records: Option<serde_json::Value>,
    /// Native file format id (e.g. `mooncake_trace`, `single_turn`).
    format: Option<String>,
    /// Sampling order (`sequential` default).
    sampling: Option<String>,
    entries: Option<u32>,
    #[serde(default, alias = "numConversations")]
    num_conversations: Option<u32>,
    /// Per-dataset sampling seed (distinct from the top-level run `randomSeed`).
    #[serde(default, alias = "randomSeed")]
    random_seed: Option<u64>,
    prompts: Option<PromptsSection>,
    /// Shared-prefix / prefix-pool policy (`synthetic.prefix_prompts`).
    #[serde(default, alias = "prefixPrompts")]
    prefix_prompts: Option<PrefixPromptsSection>,
    /// Turns-per-session distribution (multi-turn).
    turns: Option<DistFields>,
    /// Inter-turn fixed delay distribution, milliseconds (`turn_delay`).
    #[serde(default, alias = "turnDelay")]
    turn_delay: Option<DistFields>,
    /// Per-turn think-time delay ratio.
    #[serde(default, alias = "turnDelayRatio")]
    turn_delay_ratio: Option<f64>,
    /// Inter-turn delay cap, seconds (file/trace datasets).
    #[serde(default, alias = "interTurnDelayCapSeconds")]
    inter_turn_delay_cap_seconds: Option<f64>,
    /// Fetch remote image URLs and inline them at dataset generation time
    /// (`--prefetch-media-urls`); file/public datasets only.
    #[serde(default, alias = "prefetchMediaUrls")]
    prefetch_media_urls: Option<bool>,
    /// Strip repeated image content once observed within a session
    /// (`--uuid-and-strip`), single_turn only.
    #[serde(default, alias = "uuidAndStrip")]
    uuid_and_strip: Option<bool>,
    /// `baseten_trace` replay-timing knobs.
    #[serde(default, alias = "replaySpeedup")]
    replay_speedup: Option<f64>,
    #[serde(default, alias = "maxIdleGapCapSeconds")]
    max_idle_gap_cap_seconds: Option<f64>,
    #[serde(default, alias = "openLoopReplay")]
    open_loop_replay: Option<bool>,
    #[serde(default, alias = "openLoopStrict")]
    open_loop_strict: Option<bool>,
    #[serde(default, alias = "omitKvHints")]
    omit_kv_hints: Option<bool>,
    #[serde(default, alias = "forceMinTokens")]
    force_min_tokens: Option<bool>,
    /// Synthetic image generation (`synthetic.images`).
    images: Option<ImageSection>,
    /// Synthetic audio generation (`synthetic.audio`).
    audio: Option<AudioSection>,
    /// Synthetic video generation (`synthetic.video`).
    video: Option<VideoSection>,
}

#[derive(Debug, Deserialize)]
struct ImageSection {
    #[serde(default, alias = "batchSize")]
    batch_size: Option<u32>,
    width: Option<NumOrDist>,
    height: Option<NumOrDist>,
    format: Option<String>,
    source: Option<String>,
    #[serde(default, alias = "sourceSampling")]
    source_sampling: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AudioSection {
    #[serde(default, alias = "batchSize")]
    batch_size: Option<u32>,
    length: Option<NumOrDist>,
    format: Option<String>,
    /// Sample rates (raw config units; not converted, unlike the flag path).
    #[serde(default, alias = "sampleRates")]
    sample_rates: Option<Vec<f64>>,
    depths: Option<Vec<u32>>,
    channels: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct VideoSection {
    #[serde(default, alias = "batchSize")]
    batch_size: Option<u32>,
    duration: Option<f64>,
    fps: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<String>,
    codec: Option<String>,
    #[serde(default, alias = "synthType")]
    synth_type: Option<String>,
    audio: Option<VideoAudioSection>,
}

#[derive(Debug, Deserialize)]
struct VideoAudioSection {
    channels: Option<u32>,
    codec: Option<String>,
    depth: Option<u32>,
    #[serde(default, alias = "sampleRate")]
    sample_rate: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct PromptsSection {
    isl: Option<NumOrDist>,
    osl: Option<NumOrDist>,
    #[serde(alias = "batchSize")]
    batch_size: Option<u32>,
    #[serde(default, alias = "blockSize")]
    block_size: Option<u32>,
    corpus: Option<String>,
    #[serde(default, alias = "prefixReuseFraction")]
    prefix_reuse_fraction: Option<f64>,
    #[serde(default, alias = "prefixReuseRatio")]
    prefix_reuse_ratio: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct PrefixPromptsSection {
    #[serde(default, alias = "poolSize")]
    pool_size: Option<u32>,
    length: Option<u32>,
    #[serde(default, alias = "sharedSystemLength")]
    shared_system_length: Option<u32>,
    #[serde(default, alias = "userContextLength")]
    user_context_length: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerSection {
    name: Option<String>,
    revision: Option<String>,
    #[serde(default, alias = "trustRemoteCode")]
    trust_remote_code: bool,
    #[serde(default, alias = "applyChatTemplate")]
    apply_chat_template: bool,
    /// Offload tokenization to the inference server's `/tokenize` and
    /// `/detokenize` endpoints at this origin (`http://host:port`).
    #[serde(default, alias = "serverUrl")]
    server_url: Option<String>,
}

/// A flat single phase (shorthand) or a list of phases.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)] // single-run keeps one inline phase; not hot
enum Phases {
    One(PhaseSection),
    Many(Vec<PhaseSection>),
}

#[derive(Debug, Deserialize)]
struct PhaseSection {
    /// Phase name (`warmup`/`profiling` or a custom workflow id).
    name: Option<String>,
    /// Semantic role (`warmup` or `profiling`).
    kind: Option<String>,
    /// Piecewise-linear request-rate schedule (path, inline points, or shorthand array).
    #[serde(default, alias = "rateSeries")]
    rate_series: Option<serde_json::Value>,
    /// Arrival pattern (`concurrency`/`poisson`/`gamma`/`constant`/
    /// `user_centric`/`fixed_schedule`).
    #[serde(rename = "type")]
    phase_type: Option<String>,
    concurrency: Option<u32>,
    rate: Option<f64>,
    requests: Option<u64>,
    sessions: Option<u64>,
    #[serde(default, deserialize_with = "de_duration_opt")]
    duration: Option<f64>,
    #[serde(default, alias = "gracePeriod", deserialize_with = "de_duration_opt")]
    grace_period: Option<f64>,
    /// Gamma smoothness shape.
    smoothness: Option<f64>,
    /// Concurrency-ramp duration, seconds.
    #[serde(
        default,
        alias = "concurrencyRamp",
        deserialize_with = "de_duration_opt"
    )]
    concurrency_ramp: Option<f64>,
    /// Rate-ramp duration, seconds.
    #[serde(default, alias = "rateRamp", deserialize_with = "de_duration_opt")]
    rate_ramp: Option<f64>,
    /// Prefill (warmup-cache) concurrency.
    #[serde(default, alias = "prefillConcurrency")]
    prefill_concurrency: Option<u32>,
    /// Prefill-ramp duration, seconds.
    #[serde(default, alias = "prefillRamp", deserialize_with = "de_duration_opt")]
    prefill_ramp: Option<f64>,
    /// Post-send cancellation policy.
    cancellation: Option<CancellationSection>,
    /// User-centric concurrent-user count (`user_centric` phase).
    users: Option<u32>,
    /// Fixed-schedule auto-offset toggle (defaults to "no explicit offsets").
    #[serde(default, alias = "autoOffset")]
    auto_offset: Option<bool>,
    #[serde(default, alias = "startOffset")]
    start_offset: Option<i64>,
    #[serde(default, alias = "endOffset")]
    end_offset: Option<i64>,
    /// A boolean toggle or nested adaptive-scale block.
    #[serde(default, alias = "adaptiveScale")]
    adaptive_scale: AdaptiveScaleField,
    #[serde(default, alias = "adaptiveSustainDuration")]
    adaptive_sustain_duration: Option<f64>,
    #[serde(default, alias = "adaptiveAssessmentPeriod")]
    adaptive_assessment_period: Option<f64>,
    #[serde(default, alias = "adaptiveMinCompletedRequests")]
    adaptive_min_completed_requests: Option<u64>,
    #[serde(default, alias = "adaptiveControlVariable")]
    adaptive_control_variable: Option<String>,
    #[serde(default, alias = "adaptiveControlMin")]
    adaptive_control_min: Option<f64>,
    #[serde(default, alias = "adaptiveControlMax")]
    adaptive_control_max: Option<f64>,
    #[serde(default, alias = "adaptiveScaleStrategyType")]
    adaptive_scale_strategy_type: Option<String>,
    #[serde(default, alias = "adaptiveScaleStepPolicy")]
    adaptive_scale_step_policy: Option<String>,
    #[serde(default, alias = "adaptiveScaleBaseStep")]
    adaptive_scale_base_step: Option<i64>,
    #[serde(default, alias = "adaptiveScaleMaxStepMultiplier")]
    adaptive_scale_max_step_multiplier: Option<i64>,
    #[serde(default, alias = "adaptiveScaleStepPercent")]
    adaptive_scale_step_percent: Option<f64>,
    /// An explicit filter list or nested `{metric: {stat: {op: threshold}}}` map.
    #[serde(default)]
    sla: SlaField,
}

impl PhaseSection {
    /// Block-local SLA filters take precedence over phase-level filters.
    fn sla_filters(&self) -> anyhow::Result<Vec<SlaFilterSection>> {
        if let Some(block) = self.adaptive_scale.block()
            && let Some(sla) = &block.sla
        {
            return sla.filters();
        }
        self.sla.filters()
    }
}

#[derive(Debug, Deserialize)]
struct CancellationSection {
    rate: f64,
    delay: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct SlaFilterSection {
    #[serde(alias = "metricTag")]
    metric_tag: String,
    #[serde(default = "default_sla_stat")]
    stat: String,
    op: String,
    threshold: f64,
}

/// Default SLA statistic (`SLAFilter.stat` default).
fn default_sla_stat() -> String {
    "p95".to_string()
}

/// `adaptive_scale:` accepts a boolean toggle or nested block.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AdaptiveScaleField {
    Enabled(bool),
    Block(Box<AdaptiveScaleBlock>),
}

impl Default for AdaptiveScaleField {
    fn default() -> Self {
        AdaptiveScaleField::Enabled(false)
    }
}

impl AdaptiveScaleField {
    fn enabled(&self) -> bool {
        match self {
            AdaptiveScaleField::Enabled(b) => *b,
            AdaptiveScaleField::Block(b) => b.enabled(),
        }
    }

    /// The nested block, when the map form was authored.
    fn block(&self) -> Option<&AdaptiveScaleBlock> {
        match self {
            AdaptiveScaleField::Block(b) => Some(b),
            AdaptiveScaleField::Enabled(_) => None,
        }
    }
}

/// Nested `adaptive_scale:` block with control and strategy sub-maps.
#[derive(Debug, Default, Deserialize)]
struct AdaptiveScaleBlock {
    #[serde(default)]
    enabled: Option<BoolOrStr>,
    #[serde(default, alias = "controlVariable")]
    control_variable: Option<String>,
    #[serde(default, alias = "controlMin")]
    control_min: Option<f64>,
    #[serde(default, alias = "controlMax")]
    control_max: Option<f64>,
    #[serde(default, alias = "minConcurrency")]
    min_concurrency: Option<f64>,
    #[serde(default, alias = "maxConcurrency")]
    max_concurrency: Option<f64>,
    #[serde(default, alias = "assessmentPeriod")]
    assessment_period: Option<f64>,
    #[serde(default)]
    window: Option<f64>,
    #[serde(default, alias = "minCompletedRequests")]
    min_completed_requests: Option<u64>,
    #[serde(default, alias = "sustainDuration")]
    sustain_duration: Option<f64>,
    #[serde(default)]
    control: Option<ControlSub>,
    #[serde(default)]
    strategy: Option<StrategySub>,
    #[serde(default)]
    sla: Option<SlaField>,
}

impl AdaptiveScaleBlock {
    /// An omitted `enabled` field enables an authored block.
    fn enabled(&self) -> bool {
        match &self.enabled {
            None => true,
            Some(BoolOrStr::Bool(b)) => *b,
            Some(BoolOrStr::Str(s)) => matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "true" | "yes" | "on" | "1"
            ),
        }
    }

    /// Control variable (a `control.variable` sub-map wins over the scalar).
    fn control_variable(&self) -> Option<String> {
        self.control
            .as_ref()
            .and_then(|c| c.variable.clone())
            .or_else(|| self.control_variable.clone())
    }

    /// Control minimum: `control.min` > `min_concurrency` > `control_min`.
    fn control_min(&self) -> Option<f64> {
        self.control
            .as_ref()
            .and_then(|c| c.min)
            .or(self.min_concurrency)
            .or(self.control_min)
    }

    /// Control maximum: `control.max` > `max_concurrency` > `control_max`.
    fn control_max(&self) -> Option<f64> {
        self.control
            .as_ref()
            .and_then(|c| c.max)
            .or(self.max_concurrency)
            .or(self.control_max)
    }

    /// Assessment period (`assessment_period` wins over the `window` alias).
    fn assessment_period(&self) -> Option<f64> {
        self.assessment_period.or(self.window)
    }

    fn strategy_type(&self) -> Option<String> {
        self.strategy.as_ref().and_then(|s| s.strategy_type.clone())
    }

    fn step_policy(&self) -> Option<String> {
        self.strategy.as_ref().and_then(|s| s.step_policy.clone())
    }

    fn base_step(&self) -> Option<i64> {
        self.strategy.as_ref().and_then(|s| s.base_step)
    }

    fn max_step_multiplier(&self) -> Option<i64> {
        self.strategy.as_ref().and_then(|s| s.max_step_multiplier)
    }

    fn step_percent(&self) -> Option<f64> {
        self.strategy.as_ref().and_then(|s| s.step_percent)
    }
}

/// `adaptive_scale.control: {variable, min, max}` sub-map.
#[derive(Debug, Default, Deserialize)]
struct ControlSub {
    #[serde(default)]
    variable: Option<String>,
    #[serde(default)]
    min: Option<f64>,
    #[serde(default)]
    max: Option<f64>,
}

/// `adaptive_scale.strategy: {type, step_policy, …}` sub-map.
#[derive(Debug, Default, Deserialize)]
struct StrategySub {
    #[serde(default, rename = "type")]
    strategy_type: Option<String>,
    #[serde(default, alias = "stepPolicy")]
    step_policy: Option<String>,
    #[serde(default, alias = "baseStep")]
    base_step: Option<i64>,
    #[serde(default, alias = "maxStepMultiplier")]
    max_step_multiplier: Option<i64>,
    #[serde(default, alias = "stepPercent")]
    step_percent: Option<f64>,
}

/// A bool or a truthy/falsy string (`adaptive_scale.enabled`).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum BoolOrStr {
    Bool(bool),
    Str(String),
}

/// `sla:` accepts an explicit list of filters, or a nested
/// `{metric: {stat: {op: threshold}}}` map (`normalize_adaptive_sla`).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SlaField {
    List(Vec<SlaFilterSection>),
    Map(serde_json::Map<String, serde_json::Value>),
}

impl Default for SlaField {
    fn default() -> Self {
        SlaField::List(Vec::new())
    }
}

impl SlaField {
    /// Lower to concrete filters, preserving insertion order for the map form.
    fn filters(&self) -> anyhow::Result<Vec<SlaFilterSection>> {
        match self {
            SlaField::List(v) => Ok(v.clone()),
            SlaField::Map(m) => normalize_adaptive_sla(m),
        }
    }
}

/// Flatten nested SLA entries while preserving YAML map order.
fn normalize_adaptive_sla(
    sla: &serde_json::Map<String, serde_json::Value>,
) -> anyhow::Result<Vec<SlaFilterSection>> {
    let mut filters = Vec::new();
    for (metric_tag, stats) in sla {
        let stats = stats.as_object().ok_or_else(|| {
            anyhow::anyhow!("adaptive_scale.sla entries must map metric tags to stats")
        })?;
        for (stat, ops) in stats {
            let ops = ops.as_object().ok_or_else(|| {
                anyhow::anyhow!("adaptive_scale.sla stats must map operators to thresholds")
            })?;
            for (op, threshold) in ops {
                let threshold = threshold.as_f64().ok_or_else(|| {
                    anyhow::anyhow!("adaptive_scale.sla threshold must be a number")
                })?;
                filters.push(SlaFilterSection {
                    metric_tag: metric_tag.clone(),
                    stat: stat.clone(),
                    op: op.clone(),
                    threshold,
                });
            }
        }
    }
    Ok(filters)
}

fn enabled_or_config<T, U>(value: Option<EnabledOrConfig<T>>, map: impl FnOnce(T) -> U) -> Option<U>
where
    U: Default,
{
    match value {
        None | Some(EnabledOrConfig::Enabled(false)) => None,
        Some(EnabledOrConfig::Enabled(true)) => Some(U::default()),
        Some(EnabledOrConfig::Config(value)) => Some(map(value)),
    }
}

impl Benchmark {
    /// Normalize the parsed config into shared [`Inputs`].
    fn into_inputs(
        self,
        artifact_dir: Option<PathBuf>,
        random_seed: Option<u64>,
    ) -> anyhow::Result<Inputs> {
        let model_names = self.resolve_model_names()?;

        // Artifact dir precedence: the `--artifact-dir` flag, then the config's
        // `artifacts.dir`, then the default `artifacts/`.
        let config_artifact_dir = self.artifacts.as_ref().and_then(|a| a.dir.clone());

        // Transport first: DynoSim relaxes the endpoint URL/type requirements
        // (the runner opens no socket) and injects the never-dialed sentinel.
        let transport = parse_transport(self.transport.as_ref())?;
        let is_dynosim = transport.is_dynosim();
        let is_dry_run = matches!(transport, Transport::DryRun(_));

        // DynoSim defaults to its own dialect; other transports default to chat.
        let endpoint_type = self
            .endpoint
            .endpoint_type
            .clone()
            .or_else(|| is_dynosim.then(|| "dynosim".to_string()))
            .unwrap_or_else(|| "chat".to_string());
        let reset_kv_cache =
            enabled_or_config(self.endpoint.reset_kv_cache, |config| ResetKvCacheConfig {
                timeout_seconds: config.timeout_seconds,
                path: config.path,
            });
        let server_profiler = enabled_or_config(self.endpoint.server_profiler, |config| {
            ServerProfilerConfig {
                timeout_seconds: config.timeout_seconds,
                start_path: config.start_path,
                stop_path: config.stop_path,
            }
        });

        // DynoSim needs a never-dialed sentinel when no URL is authored.
        let urls = match self.endpoint.url {
            Some(u) => u.into_vec(),
            None if is_dynosim => vec!["dynosim://offline".to_string()],
            None => anyhow::bail!("endpoint.url is required"),
        };

        // Single-run: the shorthand `dataset:` or the first `datasets:` entry.
        let dataset = self
            .dataset
            .or_else(|| self.datasets.and_then(|d| d.into_iter().next()));
        let (isl, osl, batch_size, isl_block_size) = extract_prompts(dataset.as_ref());
        let prefix_reuse_fraction = dataset
            .as_ref()
            .and_then(|d| d.prompts.as_ref())
            .and_then(|p| p.prefix_reuse_fraction);
        let prefix_reuse_ratio = dataset
            .as_ref()
            .and_then(|d| d.prompts.as_ref())
            .and_then(|p| p.prefix_reuse_ratio);
        let prompt_corpus = dataset
            .as_ref()
            .and_then(|d| d.prompts.as_ref())
            .and_then(|p| p.corpus.clone());
        let num_conversations = dataset.as_ref().and_then(|d| d.num_conversations);
        let dataset_entries = dataset.as_ref().and_then(|d| d.entries);
        // Per-dataset seed is separate from the top-level run seed.
        let dataset_random_seed = dataset.as_ref().and_then(|d| d.random_seed);
        let inter_turn_delay_cap_seconds = dataset
            .as_ref()
            .and_then(|d| d.inter_turn_delay_cap_seconds);
        let prefetch_media_urls = dataset
            .as_ref()
            .and_then(|d| d.prefetch_media_urls)
            .unwrap_or(false);
        let uuid_and_strip = dataset
            .as_ref()
            .and_then(|d| d.uuid_and_strip)
            .unwrap_or(false);
        let replay_speedup = dataset.as_ref().and_then(|d| d.replay_speedup);
        let max_idle_gap_cap_seconds = dataset.as_ref().and_then(|d| d.max_idle_gap_cap_seconds);
        let open_loop_replay = dataset
            .as_ref()
            .and_then(|d| d.open_loop_replay)
            .unwrap_or(true);
        let open_loop_strict = dataset
            .as_ref()
            .and_then(|d| d.open_loop_strict)
            .unwrap_or(false);
        let omit_kv_hints = dataset
            .as_ref()
            .and_then(|d| d.omit_kv_hints)
            .unwrap_or(false);
        let force_min_tokens = dataset
            .as_ref()
            .and_then(|d| d.force_min_tokens)
            .unwrap_or(true);

        // Multi-turn (turns / inter-turn delay / think-time ratio).
        let turns = dataset
            .as_ref()
            .and_then(|d| d.turns.as_ref())
            .map(dist_from);
        let turn_delay_ms = dataset
            .as_ref()
            .and_then(|d| d.turn_delay.as_ref())
            .map(dist_from);
        let turn_delay_ratio = dataset
            .as_ref()
            .and_then(|d| d.turn_delay_ratio)
            .unwrap_or(1.0);

        // Shared-prefix / prefix-pool policy (`synthetic.prefix_prompts`).
        let prefix_prompts = dataset
            .as_ref()
            .and_then(|d| d.prefix_prompts.as_ref())
            .map(|p| crate::model::dataset::PrefixPrompts {
                shared_system_length: p.shared_system_length,
                user_context_length: p.user_context_length,
                length: p.length,
                pool_size: p.pool_size,
            });

        // YAML sample rates retain their authored units.
        let image_spec = dataset.as_ref().and_then(|d| d.images.as_ref()).map(|i| {
            crate::model::dataset::ImageSpec {
                batch_size: i.batch_size.unwrap_or(1),
                format: i.format.clone().unwrap_or_else(|| "jpeg".to_string()),
                height: i
                    .height
                    .as_ref()
                    .map(clone_num_or_dist)
                    .unwrap_or_else(load::default_media_dim),
                width: i
                    .width
                    .as_ref()
                    .map(clone_num_or_dist)
                    .unwrap_or_else(load::default_media_dim),
                source: i.source.clone().unwrap_or_else(|| "noise".to_string()),
                source_sampling: i
                    .source_sampling
                    .clone()
                    .unwrap_or_else(|| "random-with-replacement".to_string()),
            }
        });
        let audio_spec = dataset.as_ref().and_then(|d| d.audio.as_ref()).map(|a| {
            crate::model::dataset::AudioSpec {
                batch_size: a.batch_size.unwrap_or(1),
                channels: a.channels.unwrap_or(1),
                depths: a.depths.clone().unwrap_or_else(|| vec![16]),
                format: a.format.clone().unwrap_or_else(|| "wav".to_string()),
                length: a
                    .length
                    .as_ref()
                    .map(clone_num_or_dist)
                    .unwrap_or_else(load::default_media_dim),
                sample_rates: a.sample_rates.clone().unwrap_or_else(|| vec![16.0]),
            }
        });
        let video_spec = dataset.as_ref().and_then(|d| d.video.as_ref()).map(|v| {
            let va = v.audio.as_ref();
            crate::model::dataset::VideoSpec {
                audio: crate::model::dataset::VideoAudio {
                    channels: va.and_then(|a| a.channels).unwrap_or(0),
                    codec: va.and_then(|a| a.codec.clone()),
                    depth: va.and_then(|a| a.depth).unwrap_or(16),
                    sample_rate: va.and_then(|a| a.sample_rate).unwrap_or(44.1),
                },
                batch_size: v.batch_size.unwrap_or(1),
                codec: v.codec.clone().unwrap_or_else(|| "libvpx-vp9".to_string()),
                duration: v.duration.unwrap_or(1.0),
                format: v.format.clone().unwrap_or_else(|| "webm".to_string()),
                fps: v.fps.unwrap_or(4),
                synth_type: v
                    .synth_type
                    .clone()
                    .unwrap_or_else(|| "moving_shapes".to_string()),
                width: v.width,
                height: v.height,
            }
        });

        // Sampling order applies to both synthetic and file datasets.
        let sampling = dataset
            .as_ref()
            .and_then(|d| d.sampling.clone())
            .unwrap_or_else(|| "sequential".to_string());

        // Dataset kind: `public` (catalog name) > `file` (path/format) >
        // synthetic. `build()` picks public first when `public_dataset` is set.
        let dataset_type = dataset.as_ref().and_then(|d| d.dataset_type.as_deref());
        // Fail closed on an unrecognized dataset type rather than silently
        // treating it as synthetic (Config's discriminated union rejects it too).
        if let Some(t) = dataset_type
            && !matches!(t, "synthetic" | "file" | "public")
        {
            anyhow::bail!("unknown dataset.type {t:?} (expected synthetic/file/public)");
        }
        let public_dataset = (dataset_type == Some("public"))
            .then(|| dataset.as_ref().and_then(|d| d.public_name.clone()))
            .flatten();
        let hf_dataset = dataset.as_ref().and_then(|d| d.hf_dataset.clone());
        anyhow::ensure!(
            dataset_type != Some("public") || public_dataset.is_some() || hf_dataset.is_some(),
            "dataset.type=public requires a `dataset:` catalog name or `hf_dataset:`"
        );
        let hf_subset = dataset.as_ref().and_then(|d| d.hf_subset.clone());
        let hf_split = dataset.as_ref().and_then(|d| d.hf_split.clone());
        let hf_revision = dataset.as_ref().and_then(|d| d.hf_revision.clone());
        let hf_text_column = dataset.as_ref().and_then(|d| d.hf_text_column.clone());
        let hf_output_column = dataset.as_ref().and_then(|d| d.hf_output_column.clone());
        let hf_output_len = dataset.as_ref().and_then(|d| d.hf_output_len);
        let hf_format = (dataset_type == Some("public"))
            .then(|| dataset.as_ref().and_then(|d| d.format.clone()))
            .flatten();
        let is_file = dataset_type == Some("file");
        let (input_file, inline_records, custom_dataset_type) = if is_file {
            let d = dataset.as_ref().expect("file dataset present");
            // A file dataset is either path-backed or carries inline `records:`
            // authored directly in the config (mutually exclusive).
            anyhow::ensure!(
                d.path.is_some() || d.records.is_some(),
                "dataset.type=file requires a `path:` or inline `records:`"
            );
            anyhow::ensure!(
                !(d.path.is_some() && d.records.is_some()),
                "dataset.type=file cannot set both `path:` and inline `records:`"
            );
            (
                d.path.clone().map(PathBuf::from),
                d.records.clone(),
                d.format.clone(),
            )
        } else {
            (None, None, None)
        };

        // Multi-phase YAML lists are preserved verbatim; legacy single-phase axes still
        // derive from the last non-warmup entry for CLI overlay fields.
        let phases_override = match &self.phases {
            Some(Phases::Many(v)) => Some(
                v.iter()
                    .map(yaml_phase_to_model)
                    .collect::<anyhow::Result<Vec<_>>>()?,
            ),
            _ => None,
        };

        // The profiling phase comes from `phases:` (flat or list) or the simple
        // `profiling:` block; the two forms are mutually exclusive (as in Config).
        // An explicit `phases:` list routes by name: the `warmup`-named entry
        // becomes the run's warmup axes, everything else is the profiling phase.
        let (phase, list_warmup) = match (self.phases, self.profiling) {
            (Some(_), Some(_)) => {
                anyhow::bail!("'phases' cannot be combined with 'warmup'/'profiling'")
            }
            (Some(Phases::One(p)), None) => (p, None),
            (Some(Phases::Many(v)), None) => {
                let mut warmup = None;
                let mut profiling = None;
                for p in v {
                    if p.name.as_deref() == Some("warmup") || p.kind.as_deref() == Some("warmup") {
                        warmup = Some(p);
                    } else {
                        // Last non-warmup entry wins (a single profiling phase).
                        profiling = Some(p);
                    }
                }
                let profiling = profiling
                    .ok_or_else(|| anyhow::anyhow!("phases must include a non-warmup phase"))?;
                (profiling, warmup)
            }
            (None, Some(p)) => (p, None),
            (None, None) => anyhow::bail!("a phase is required (set `phases` or `profiling`)"),
        };

        // A leading `warmup:` block (simple-config form) OR a `warmup`-named entry
        // in the `phases:` list becomes the run's warmup axes, excluded from
        // results and run before profiling.
        let warmup = self.warmup.or(list_warmup).map(|w| Warmup {
            concurrency: w.concurrency,
            rate: w.rate,
            requests: w.requests,
            sessions: w.sessions,
            prefill_concurrency: w.prefill_concurrency,
            rate_mode: w
                .phase_type
                .as_deref()
                .filter(|t| matches!(*t, "poisson" | "gamma" | "constant"))
                .map(str::to_string),
            concurrency_ramp: w.concurrency_ramp,
            rate_ramp: w.rate_ramp,
            prefill_ramp: w.prefill_ramp,
            duration: w.duration,
            grace_period: w.grace_period,
        });

        // Phase arrival pattern. A rate `type` selects the arrival distribution;
        // `user_centric` binds (rate, users); `concurrency` (default) has no rate.
        let phase_type = phase.phase_type.as_deref();
        // Fail closed on an unknown phase type rather than silently defaulting to
        // concurrency (Config's discriminated phase union rejects it too).
        if let Some(t) = phase_type
            && !matches!(
                t,
                "concurrency"
                    | "poisson"
                    | "gamma"
                    | "constant"
                    | "user_centric"
                    | "fixed_schedule"
            )
        {
            anyhow::bail!(
                "unknown phase type {t:?} (expected concurrency/poisson/gamma/constant/user_centric/fixed_schedule)"
            );
        }
        let is_user_centric = phase_type == Some("user_centric");
        let rate_mode = match phase_type {
            Some(t @ ("poisson" | "gamma" | "constant")) => Some(t.to_string()),
            _ => None,
        };
        let user_centric = match (is_user_centric, phase.rate, phase.users) {
            (true, Some(rate), Some(users)) => Some((rate, users)),
            (true, _, _) => anyhow::bail!("user_centric phase requires rate and users"),
            _ => None,
        };
        // user_centric drives its own rate; keep request_rate clear for it.
        let phase_rate = if is_user_centric { None } else { phase.rate };
        let phase_cancellation = phase.cancellation.as_ref().map(|c| (c.rate, c.delay));

        // Configured schedules leave request counting to the runner.
        let is_fixed_schedule = phase_type == Some("fixed_schedule");
        let fixed_schedule = is_fixed_schedule.then(|| {
            phase
                .auto_offset
                .unwrap_or(phase.start_offset.is_none() && phase.end_offset.is_none())
        });

        // Adaptive scale (opt-in via the phase's `adaptive_scale: true` toggle
        // or an enabled nested block).
        let adaptive_scale = if phase.adaptive_scale.enabled() {
            Some(build_adaptive_yaml(&phase)?)
        } else {
            None
        };

        // Synthetic conversation count never derives from the request bound.
        let entries = num_conversations
            .or(dataset_entries)
            .unwrap_or(load::DEFAULT_ENTRIES);

        let (
            tokenizer_name,
            tokenizer_revision,
            tokenizer_trust,
            apply_chat_template,
            server_tokenizer_url,
        ) = match self.tokenizer {
            Some(t) => (
                t.name,
                t.revision,
                t.trust_remote_code,
                t.apply_chat_template,
                t.server_url,
            ),
            None => (None, None, false, false, None),
        };

        // Model-selection strategy (`models.strategy`).
        let model_strategy = self
            .models
            .as_ref()
            .and_then(|m| m.strategy.as_deref())
            .map(load::parse_model_strategy)
            .transpose()?;

        // Goodput SLOs (`benchmark.slos`): metric -> ms as a JSON map.
        let slos: serde_json::Map<String, serde_json::Value> = self
            .slos
            .as_ref()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), serde_json::json!(v)))
                    .collect()
            })
            .unwrap_or_default();

        // GPU telemetry (default enabled): optional custom DCGM URLs.
        let (gpu_enabled, gpu_urls) = self
            .gpu_telemetry
            .as_ref()
            .map(|g| {
                (
                    g.enabled.unwrap_or(true),
                    g.urls.clone().unwrap_or_default(),
                )
            })
            .unwrap_or((true, Vec::new()));

        // Server metrics (default enabled): optional scrape URLs and formats.
        let (sm_enabled, sm_urls, sm_formats) = self
            .server_metrics
            .as_ref()
            .map(|s| {
                (
                    s.enabled.unwrap_or(true),
                    s.urls.clone().unwrap_or_default(),
                    s.formats.clone(),
                )
            })
            .unwrap_or((true, Vec::new(), None));

        // A fixed network-latency mean takes precedence over probing.
        let (network_latency_mean, network_latency_probe) = match self.network_latency.as_ref() {
            Some(nl) if nl.mean_ms.is_some() => (nl.mean_ms, None),
            Some(nl) if nl.enabled => (None, Some(nl.ping_interval.unwrap_or(1.0))),
            _ => (None, None),
        };

        let otel_url = self.otel.as_ref().and_then(|o| o.metrics_url.clone());
        // MLflow's `total_expected_requests` is the run's request bound.
        let total_expected_requests = phase.requests.map(|n| n as f64);
        let mlflow = self
            .mlflow
            .as_ref()
            .map(|m| crate::model::export::MlflowParams {
                tracking_uri: m.tracking_uri.clone(),
                experiment: m.experiment.clone(),
                run_name: m.run_name.clone(),
                parent_run_id: None,
                tags: Vec::new(),
                artifact_globs: Vec::new(),
                total_expected_requests,
            })
            .unwrap_or(crate::model::export::MlflowParams {
                tracking_uri: None,
                experiment: None,
                run_name: None,
                parent_run_id: None,
                tags: Vec::new(),
                artifact_globs: Vec::new(),
                total_expected_requests,
            });
        let wandb = self
            .wandb
            .as_ref()
            .map(|w| crate::model::export::WandbParams {
                project: w.project.clone(),
                entity: w.entity.clone(),
                run_name: w.run_name.clone(),
                tags: Vec::new(),
            })
            .unwrap_or(crate::model::export::WandbParams {
                project: None,
                entity: None,
                run_name: None,
                tags: Vec::new(),
            });

        // Runtime worker/cell/dispatch/hop-routing policy.
        let (
            runtime_workers,
            runtime_workers_min,
            runtime_cells,
            runtime_dispatch,
            runtime_hop_routing,
        ) = self
            .runtime
            .as_ref()
            .map(|r| {
                (
                    r.workers,
                    r.workers_min,
                    r.cells.unwrap_or(1),
                    r.dispatch,
                    r.hop_routing,
                )
            })
            .unwrap_or((None, None, 1, None, None));

        // Timeslice window (`artifacts.slice_duration`).
        let slice_duration = self.artifacts.as_ref().and_then(|a| a.slice_duration);

        // Per-record export: `artifacts.records` (default JSONL), `raw`, `trace`,
        // `export_outputs_json`.
        let records_formats = match self.artifacts.as_ref().and_then(|a| a.records.as_ref()) {
            Some(RecordsFormats::List(v)) => v.clone(),
            Some(RecordsFormats::Disabled(_)) => Vec::new(),
            None => vec!["jsonl".to_string()],
        };
        let export_raw = self.artifacts.as_ref().is_some_and(|a| a.raw);
        let show_trace_timing = self
            .artifacts
            .as_ref()
            .is_some_and(|a| a.show_trace_timing);
        let export_trace = self.artifacts.as_ref().is_some_and(|a| a.trace) || show_trace_timing;
        let export_outputs_json = self
            .artifacts
            .as_ref()
            .is_some_and(|a| a.export_outputs_json);

        // Multi-phase YAML already owns every phase axis in `phases_override`.
        // Mirrored scalars from the last non-warmup entry must not look like
        // CLI loadgen overlays (that false-positive fails closed when more than
        // one profiling phase exists). Explicit `--concurrency`/etc. still flow
        // through `apply_cli_overrides` onto Inputs.
        let multiphase_authored = phases_override.is_some();

        Ok(Inputs {
            model_names,
            urls,
            endpoint_type,
            transport,
            streaming: self.endpoint.streaming,
            timeout_seconds: self.endpoint.timeout,
            use_legacy_max_tokens: self.endpoint.use_legacy_max_tokens.unwrap_or(false),
            use_server_token_count: self.endpoint.use_server_token_count.unwrap_or(false),
            download_video_content: self.endpoint.download_video_content.unwrap_or(false),
            extra: self.endpoint.extra.unwrap_or_default(),
            server_metrics_urls: sm_urls,
            connection_reuse: self
                .endpoint
                .connection_reuse
                .as_deref()
                .map(load::parse_connection_reuse)
                .transpose()?,
            request_content_type: self
                .endpoint
                .request_content_type
                .as_deref()
                .map(load::parse_content_type)
                .transpose()?,
            wait_for_model_timeout: self.endpoint.wait_for_model_timeout,
            wait_for_model_mode: self
                .endpoint
                .wait_for_model_mode
                .as_deref()
                .map(load::parse_wait_mode)
                .transpose()?,
            wait_for_model_interval: self.endpoint.wait_for_model_interval,
            apply_chat_template,
            prefill_concurrency: if multiphase_authored {
                None
            } else {
                phase.prefill_concurrency
            },
            prefill_ramp: if multiphase_authored {
                None
            } else {
                phase.prefill_ramp
            },
            gpu_telemetry_enabled: gpu_enabled,
            gpu_telemetry_urls: gpu_urls,
            gpu_telemetry_metrics_file: None,
            server_metrics_enabled: sm_enabled,
            server_metrics_formats: sm_formats,
            slos,
            network_latency_mean,
            network_latency_probe,
            otel_url,
            otel_provider: None,
            otel_resource_attributes: Vec::new(),
            mlflow,
            wandb,
            api_key: self.endpoint.api_key,
            headers: self.endpoint.headers.unwrap_or_default(),
            tokenizer_name,
            tokenizer_revision,
            tokenizer_trust,
            server_tokenizer_url,
            isl,
            osl,
            turns,
            turn_delay_ratio,
            turn_delay_ms,
            session_header: self.endpoint.session_header,
            proxy: self.endpoint.proxy,
            proxy_from_env: self.endpoint.proxy_from_env.unwrap_or(false),
            endpoint_path: self.endpoint.path,
            reset_kv_cache,
            server_profiler,
            records_formats,
            export_raw,
            export_trace,
            export_outputs_json,
            show_trace_timing,
            profile_export_prefix: self.artifacts.as_ref().and_then(|a| a.prefix.clone()),
            use_think_time_only: false,
            max_context_length: None,
            allow_dataset_wrap: None,
            cache_bust: None,
            burst_phase_starts: false,
            trace_idle_gap_cap_seconds: None,
            system_idle_gap_cap_seconds: None,
            hf_weka_dataset: None,
            trace_session_sample_ratio: None,
            agentic_warmup_grace_period: None,
            failed_request_threshold: None,
            sequence_distribution: None,
            batch_size: batch_size.unwrap_or(1),
            sampling,
            entries,
            // Explicit entries only (file/public); synthetic uses `entries`.
            dataset_entries,
            sessions: if multiphase_authored {
                None
            } else {
                num_conversations.map(u64::from).or(phase.sessions)
            },
            concurrency: if multiphase_authored {
                None
            } else {
                phase.concurrency
            },
            request_rate: if multiphase_authored {
                None
            } else {
                phase_rate
            },
            rate_mode: if multiphase_authored { None } else { rate_mode },
            smoothness: if multiphase_authored {
                None
            } else {
                phase.smoothness
            },
            concurrency_ramp: if multiphase_authored {
                None
            } else {
                phase.concurrency_ramp
            },
            rate_ramp: if multiphase_authored {
                None
            } else {
                phase.rate_ramp
            },
            cancellation: if multiphase_authored {
                None
            } else {
                phase_cancellation
            },
            user_centric: if multiphase_authored {
                None
            } else {
                user_centric
            },
            request_count: if multiphase_authored {
                None
            } else {
                phase.requests
            },
            benchmark_duration: if multiphase_authored {
                None
            } else {
                phase.duration
            },
            grace_period: if multiphase_authored {
                None
            } else {
                phase.grace_period
            },
            warmup,
            runtime_workers,
            runtime_workers_min,
            runtime_cells,
            runtime_dispatch,
            runtime_hop_routing,
            random_seed,
            dataset_random_seed,
            input_file,
            inline_records,
            custom_dataset_type,
            public_dataset,
            hf_subset,
            hf_dataset,
            hf_split,
            hf_revision,
            hf_text_column,
            hf_output_column,
            hf_output_len,
            hf_format,
            inter_turn_delay_cap_seconds,
            prefetch_media_urls,
            uuid_and_strip,
            replay_speedup,
            max_idle_gap_cap_seconds,
            open_loop_replay,
            open_loop_strict,
            omit_kv_hints,
            force_min_tokens,
            fixed_schedule,
            fixed_schedule_start_offset: phase.start_offset,
            fixed_schedule_end_offset: phase.end_offset,
            model_strategy,
            slice_duration,
            isl_block_size,
            prefix_reuse_fraction,
            prefix_reuse_ratio,
            prompt_corpus,
            sketch_metrics: false,
            steady_state: false,
            steady_state_fraction: None,
            steady_state_hybrid: false,
            image_spec,
            audio_spec,
            video_spec,
            adaptive_scale,
            prefix_prompts,
            // Authored via `benchmark.scenario` / `.trajectoryStart*` / `.unsafeOverride`.
            scenario: self.scenario.clone(),
            // The YAML config path selects semantics via the scenario (derived in
            // `build`); the explicit override is the `--weka-semantics` CLI flag.
            weka_semantics: None,
            // No YAML surface yet; the CLI `--ignore-trace-delays` flag is the
            // only authoring path. Default to honoring recorded trace delays.
            ignore_trace_delays: false,
            ignore_trace_delays_explicit: false,
            trajectory_start_min_ratio: self.trajectory_start_min_ratio.unwrap_or(0.0),
            trajectory_start_max_ratio: self.trajectory_start_max_ratio.unwrap_or(0.0),
            unsafe_override: self.unsafe_override.unwrap_or(false),
            agentic_cache_warmup_duration: None,
            rankings: None,
            accuracy: None,
            synthesis: None,
            dataset_filters: None,
            // A `dry_run` config transport emits the dataset-analysis family with
            // default knobs; `apply_cli_overrides` layers `--kv-*` /
            // `--no-dataset-analysis` on top.
            dataset_analysis: is_dry_run.then(|| load::DatasetAnalysisInputs {
                block_size: 16,
                cache_blocks: None,
                per_conversation: false,
            }),
            phases_override,
            request_rate_series: if multiphase_authored {
                None
            } else {
                phase
                    .rate_series
                    .as_ref()
                    .map(parse_yaml_rate_series)
                    .transpose()?
            },
            artifact_dir: artifact_dir
                .or_else(|| config_artifact_dir.map(PathBuf::from))
                .unwrap_or_else(|| PathBuf::from("artifacts")),
        })
    }

    /// Resolve the model list from the `model:` shorthand or `models:` block.
    fn resolve_model_names(&self) -> anyhow::Result<Vec<String>> {
        if let Some(models) = &self.models {
            let names: Vec<String> = models.items.iter().map(|m| m.name.clone()).collect();
            anyhow::ensure!(!names.is_empty(), "models.items must not be empty");
            return Ok(names);
        }
        match &self.model {
            Some(m) => Ok(clone_string_or_vec(m)),
            None => anyhow::bail!("a model is required (set `model:` or `models:`)"),
        }
    }
}

/// Map a YAML `transport.type` string to the typed [`Transport`] (default HTTP).
fn parse_transport(section: Option<&TransportSection>) -> anyhow::Result<Transport> {
    let Some(section) = section else {
        return Ok(Transport::Http);
    };
    let dynosim = || {
        let mut cfg: DynosimConfig = section.dynosim.clone();
        cfg.normalize();
        cfg
    };
    Ok(match section.transport_type.as_str() {
        "http" => Transport::Http,
        "grpc" => Transport::Grpc,
        "dynosim_offline" => Transport::DynosimOffline(dynosim()),
        "dynosim_online" => Transport::DynosimOnline(dynosim()),
        "dry_run" => Transport::DryRun(section.dry_run.clone()),
        other => anyhow::bail!("unknown transport.type {other:?}"),
    })
}

/// Extract the ISL distribution, optional OSL, batch size, and block size.
fn extract_prompts(
    dataset: Option<&DatasetSection>,
) -> (Distribution, Option<Distribution>, Option<u32>, Option<u32>) {
    let Some(prompts) = dataset.and_then(|d| d.prompts.as_ref()) else {
        return (default_isl(), None, None, None);
    };
    let isl = match &prompts.isl {
        Some(n) => clone_num_or_dist(n),
        None => default_isl(),
    };
    let osl = prompts.osl.as_ref().map(clone_num_or_dist);
    (isl, osl, prompts.batch_size, prompts.block_size)
}

fn parse_yaml_rate_series(
    value: &serde_json::Value,
) -> anyhow::Result<crate::model::rate_series::RateSeries> {
    match value {
        serde_json::Value::String(path) => {
            crate::model::rate_series::RateSeries::from_json_path(path)
        }
        other => {
            let text = serde_json::to_string(other)?;
            crate::model::rate_series::RateSeries::from_json_str(&text)
        }
    }
}

fn yaml_phase_role(
    kind: Option<&str>,
    name: &str,
) -> anyhow::Result<crate::model::phase::PhaseRole> {
    if let Some(kind) = kind {
        return match kind {
            "warmup" => Ok(crate::model::phase::PhaseRole::Warmup),
            "profiling" => Ok(crate::model::phase::PhaseRole::Profiling),
            other => anyhow::bail!("unknown phase kind {other:?}"),
        };
    }
    match name {
        "warmup" => Ok(crate::model::phase::PhaseRole::Warmup),
        "profiling" => Ok(crate::model::phase::PhaseRole::Profiling),
        _ => anyhow::bail!("phase {name:?} requires explicit kind (warmup or profiling)"),
    }
}

fn yaml_phase_to_model(section: &PhaseSection) -> anyhow::Result<crate::model::phase::Phase> {
    use crate::model::phase::{Cancellation, Phase, PhaseCommon, PhaseKind, PhaseRole, Ramp};
    let name = section
        .name
        .clone()
        .unwrap_or_else(|| "profiling".to_string());
    let role = yaml_phase_role(section.kind.as_deref(), &name)?;
    let phase_type = section.phase_type.as_deref();
    if let Some(t) = phase_type
        && !matches!(
            t,
            "concurrency" | "poisson" | "gamma" | "constant" | "user_centric" | "fixed_schedule"
        )
    {
        anyhow::bail!("unknown phase type {t:?}");
    }
    let rate_mode = match phase_type {
        Some(t @ ("poisson" | "gamma" | "constant")) => Some(t.to_string()),
        _ => None,
    };
    let mut rate = if phase_type == Some("user_centric") {
        None
    } else {
        section.rate
    };
    let rate_series = section
        .rate_series
        .as_ref()
        .map(parse_yaml_rate_series)
        .transpose()?;
    if rate.is_some() && rate_series.is_some() {
        anyhow::bail!("rate and rate_series are mutually exclusive");
    }
    if let Some(series) = &rate_series {
        rate = Some(series.initial_qps());
    }
    let kind = if phase_type == Some("user_centric") {
        let rate = section
            .rate
            .ok_or_else(|| anyhow::anyhow!("user_centric requires rate"))?;
        let users = section
            .users
            .ok_or_else(|| anyhow::anyhow!("user_centric requires users"))?;
        PhaseKind::UserCentric {
            rate,
            users,
            concurrency: section.concurrency,
        }
    } else if phase_type == Some("fixed_schedule") {
        PhaseKind::FixedSchedule {
            auto_offset: section
                .auto_offset
                .unwrap_or(section.start_offset.is_none() && section.end_offset.is_none()),
            start_offset: section.start_offset,
            end_offset: section.end_offset,
        }
    } else {
        let default_concurrency = section.concurrency.unwrap_or(1);
        if let Some(rate) = rate {
            match rate_mode.as_deref() {
                Some("gamma") => PhaseKind::Gamma {
                    rate,
                    concurrency: section.concurrency,
                    smoothness: section.smoothness,
                },
                Some("constant") => PhaseKind::Constant {
                    rate,
                    concurrency: section.concurrency,
                },
                _ => PhaseKind::Poisson {
                    rate,
                    concurrency: section.concurrency,
                },
            }
        } else {
            PhaseKind::Concurrency {
                concurrency: default_concurrency,
            }
        }
    };
    let adaptive_scale = if section.adaptive_scale.enabled() {
        Some(build_adaptive_yaml(section)?)
    } else {
        None
    };
    Ok(Phase {
        common: PhaseCommon {
            timing_mode: None,
            name,
            kind: Some(role),
            exclude_from_results: role == PhaseRole::Warmup,
            seamless: false,
            requests: section.requests,
            sessions: section.sessions,
            duration: section.duration,
            prefill_concurrency: section.prefill_concurrency,
            grace_period: section.grace_period,
            concurrency_ramp: section.concurrency_ramp.map(|duration| Ramp {
                duration,
                strategy: "linear".into(),
            }),
            prefill_ramp: section.prefill_ramp.map(|duration| Ramp {
                duration,
                strategy: "linear".into(),
            }),
            rate_ramp: section.rate_ramp.map(|duration| Ramp {
                duration,
                strategy: "linear".into(),
            }),
            cancellation: section.cancellation.as_ref().map(|c| Cancellation {
                rate: c.rate,
                delay: c.delay,
            }),
            agentic_cache_warmup_duration: None,
            agentic_warmup_grace_period: None,
            failed_request_threshold: None,
            adaptive_scale,
            rate_series,
        },
        kind,
    })
}

/// Build adaptive scaling while preserving explicit float and derived integer bounds.
fn build_adaptive_yaml(phase: &PhaseSection) -> anyhow::Result<crate::model::phase::AdaptiveScale> {
    use crate::model::phase::{AdaptiveScale, SlaFilter};
    // Nested adaptive-scale values take precedence over flat fields.
    let block = phase.adaptive_scale.block();
    let eff_control_variable = block
        .and_then(|b| b.control_variable())
        .or_else(|| phase.adaptive_control_variable.clone());
    let eff_control_min = block
        .and_then(|b| b.control_min())
        .or(phase.adaptive_control_min);
    let eff_control_max = block
        .and_then(|b| b.control_max())
        .or(phase.adaptive_control_max);
    let eff_assessment = block
        .and_then(|b| b.assessment_period())
        .or(phase.adaptive_assessment_period);
    let eff_min_completed = block
        .and_then(|b| b.min_completed_requests)
        .or(phase.adaptive_min_completed_requests);
    let eff_sustain = block
        .and_then(|b| b.sustain_duration)
        .or(phase.adaptive_sustain_duration);
    let eff_strategy_type = block
        .and_then(|b| b.strategy_type())
        .or_else(|| phase.adaptive_scale_strategy_type.clone());
    let eff_step_policy = block
        .and_then(|b| b.step_policy())
        .or_else(|| phase.adaptive_scale_step_policy.clone());
    let eff_base_step = block
        .and_then(|b| b.base_step())
        .or(phase.adaptive_scale_base_step);
    let eff_max_step_multiplier = block
        .and_then(|b| b.max_step_multiplier())
        .or(phase.adaptive_scale_max_step_multiplier);
    let eff_step_percent = block
        .and_then(|b| b.step_percent())
        .or(phase.adaptive_scale_step_percent);

    let sla_filters = phase.sla_filters()?;

    let sustain = eff_sustain
        .ok_or_else(|| anyhow::anyhow!("adaptive_scale requires adaptive_sustain_duration"))?;
    anyhow::ensure!(
        !sla_filters.is_empty(),
        "adaptive_scale requires at least one sla filter"
    );
    let control_variable = eff_control_variable.unwrap_or_else(|| "concurrency".to_string());
    // Explicit config bound -> float; else the control axis value -> int.
    let float_num =
        |v: f64| serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0));
    let minimum = match eff_control_min {
        Some(v) => float_num(v),
        None => serde_json::Number::from(1i64),
    };
    let axis_default = match control_variable.as_str() {
        "request_rate" => phase.rate.map(float_num),
        "users" => phase.users.map(|u| serde_json::Number::from(i64::from(u))),
        "prefill_concurrency" => phase
            .prefill_concurrency
            .map(|c| serde_json::Number::from(i64::from(c))),
        _ => phase
            .concurrency
            .map(|c| serde_json::Number::from(i64::from(c))),
    };
    let maximum = match eff_control_max {
        Some(v) => float_num(v),
        None => axis_default
            .ok_or_else(|| anyhow::anyhow!("adaptive_scale could not resolve a maximum"))?,
    };
    Ok(AdaptiveScale {
        control_variable,
        minimum,
        maximum,
        assessment_period_seconds: eff_assessment.unwrap_or(30.0),
        sustain_duration_seconds: sustain,
        min_completed_requests: eff_min_completed.unwrap_or(1),
        strategy_type: eff_strategy_type.unwrap_or_else(|| "ramp_until_fail".to_string()),
        step_policy: eff_step_policy.unwrap_or_else(|| "sla_margin".to_string()),
        base_step: eff_base_step.unwrap_or(10),
        max_step_multiplier: eff_max_step_multiplier.unwrap_or(4),
        step_percent: eff_step_percent.unwrap_or(25.0),
        sla_filters: sla_filters
            .iter()
            .map(|s| SlaFilter {
                metric_tag: s.metric_tag.clone(),
                stat: s.stat.clone(),
                op: s.op.clone(),
                threshold: s.threshold,
            })
            .collect(),
    })
}

/// Build a [`Distribution`] from a parametric YAML dist block (`{mean,stddev,…}`),
/// applying the config discriminator's normal-distribution default.
fn dist_from(d: &DistFields) -> Distribution {
    normalize_dist(Distribution {
        value: d.value,
        mean: d.mean,
        stddev: d.stddev,
        median: d.median,
        min: d.min,
        max: d.max,
        ..Default::default()
    })
}

/// Default bare `{mean}` distributions to zero standard deviation.
fn normalize_dist(mut d: Distribution) -> Distribution {
    if d.mean.is_some()
        && d.stddev.is_none()
        && d.median.is_none()
        && d.peaks.is_none()
        && d.value.is_none()
    {
        d.stddev = Some(0.0);
    }
    d
}

/// Clone a `StringOrVec` into a `Vec<String>` without consuming it.
fn clone_string_or_vec(v: &StringOrVec) -> Vec<String> {
    match v {
        StringOrVec::One(s) => vec![s.clone()],
        StringOrVec::Many(list) => list.clone(),
    }
}

/// Clone a `NumOrDist` into a `Distribution` without consuming it.
fn clone_num_or_dist(n: &NumOrDist) -> Distribution {
    match n {
        NumOrDist::Num(value) => Distribution {
            value: Some(*value),
            ..Default::default()
        },
        NumOrDist::Dist(d) => dist_from(d),
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve_expanded_value, resolve_str};

    /// A minimal valid config with the given `dataset:`/`phases:` bodies spliced in.
    fn cfg(body: &str) -> String {
        format!(
            "schemaVersion: \"2.0\"\n\
             benchmark:\n\
             \x20 model: m\n\
             \x20 endpoint: {{type: chat, url: 127.0.0.1:8000}}\n\
             {body}"
        )
    }

    fn err(body: &str) -> String {
        resolve_str(&cfg(body), Some("/tmp/x".into()))
            .expect_err("expected a validation error")
            .to_string()
    }

    #[test]
    fn rejects_unknown_dataset_type() {
        let e = err(
            "  dataset: {type: bogus}\n  phases: {type: concurrency, requests: 1, concurrency: 1}\n",
        );
        assert!(e.contains("unknown dataset.type"), "{e}");
    }

    #[test]
    fn rejects_file_dataset_without_path() {
        let e = err(
            "  dataset: {type: file, format: mooncake_trace}\n  phases: {type: concurrency, requests: 1, concurrency: 1}\n",
        );
        assert!(e.contains("requires a `path:`"), "{e}");
    }

    #[test]
    fn rejects_public_dataset_without_name() {
        let e = err(
            "  dataset: {type: public}\n  phases: {type: concurrency, requests: 1, concurrency: 1}\n",
        );
        assert!(e.contains("requires a `dataset:` catalog name"), "{e}");
    }

    #[test]
    fn accepts_public_type_with_hf_dataset_and_no_catalog_name() {
        let run = resolve_str(
            &cfg(
                "  dataset: {type: public, hf_dataset: allenai/WildChat, hf_split: train}\n  phases: {type: concurrency, requests: 2, concurrency: 1}\n",
            ),
            Some("/tmp/x".into()),
        )
        .expect("valid hf public config resolves");
        let v = serde_json::to_value(&run).unwrap();
        let ds = &v["cfg"]["datasets"][0];
        assert_eq!(ds["type"], serde_json::json!("public"));
        assert_eq!(ds["name"], serde_json::json!("allenai/WildChat"));
        assert_eq!(ds["format"], serde_json::json!("hf"));
        assert_eq!(ds["source"]["type"], serde_json::json!("hugging_face"));
        assert_eq!(
            ds["source"]["dataset"],
            serde_json::json!("allenai/WildChat")
        );
        assert_eq!(ds["source"]["split"], serde_json::json!("train"));
    }

    #[test]
    fn rejects_baseten_extra_input_collision_from_yaml() {
        // The baseten_trace loader injects `min_tokens` per-turn from the recorded
        // output length, silently clobbering an endpoint-level `extra.min_tokens`.
        // This guard must fire on the YAML `endpoint.extra` path exactly as it does
        // on the `--extra-inputs` flags path (shared through `load::build`).
        let cfg = "schemaVersion: \"2.0\"\n\
             benchmark:\n\
             \x20 model: m\n\
             \x20 endpoint: {type: chat, url: 127.0.0.1:8000, extra: {min_tokens: 5}}\n\
             \x20 dataset: {type: file, format: baseten_trace, path: /tmp/x.jsonl}\n\
             \x20 phases: {type: concurrency, requests: 1, concurrency: 1}\n";
        let e = resolve_str(cfg, Some("/tmp/x".into()))
            .expect_err("expected a baseten extra-input collision error")
            .to_string();
        assert!(e.contains("overwritten per-turn"), "{e}");
        assert!(e.contains("min_tokens"), "{e}");
    }

    #[test]
    fn rejects_unknown_phase_type() {
        let e = err("  dataset: {prompts: {isl: 128}}\n  phases: {type: bogus, requests: 1}\n");
        assert!(e.contains("unknown phase type"), "{e}");
    }

    #[test]
    fn rejects_phases_with_profiling() {
        let e = err(
            "  dataset: {prompts: {isl: 128}}\n  phases: {type: concurrency, requests: 1, concurrency: 1}\n  profiling: {type: concurrency, requests: 1, concurrency: 1}\n",
        );
        assert!(e.contains("cannot be combined"), "{e}");
    }

    #[test]
    fn accepts_minimal_synthetic() {
        let run = resolve_str(
            &cfg("  dataset: {prompts: {isl: 128}}\n  phases: {type: concurrency, requests: 2, concurrency: 1}\n"),
            Some("/tmp/x".into()),
        )
        .expect("valid config resolves");
        let v = serde_json::to_value(&run).unwrap();
        assert_eq!(
            v["cfg"]["datasets"][0]["type"],
            serde_json::json!("synthetic")
        );
    }

    #[test]
    fn synthetic_prompt_corpus_is_yaml_authorable() {
        let run = resolve_str(
            &cfg(
                "  dataset: {prompts: {isl: 128, corpus: coding}}\n  phases: {type: concurrency, requests: 2, concurrency: 1}\n",
            ),
            Some("/tmp/x".into()),
        )
        .expect("valid config resolves");
        let v = serde_json::to_value(&run).unwrap();
        assert_eq!(
            v["cfg"]["datasets"][0]["prompts"]["corpus"],
            serde_json::json!("coding")
        );
    }

    #[test]
    fn file_prompt_corpus_is_yaml_authorable() {
        let run = resolve_str(
            &cfg(
                "  dataset: {type: file, format: mooncake_trace, path: trace.jsonl, prompts: {corpus: random}}\n  phases: {type: concurrency, requests: 2, concurrency: 1}\n",
            ),
            Some("/tmp/x".into()),
        )
        .expect("valid file config resolves");
        let v = serde_json::to_value(&run).unwrap();
        assert_eq!(
            v["cfg"]["datasets"][0]["prompts"]["corpus"],
            serde_json::json!("random")
        );
    }

    #[test]
    fn public_prompt_corpus_is_yaml_authorable() {
        let run = resolve_str(
            &cfg(
                "  dataset: {type: public, dataset: sharegpt, prompts: {corpus: coding}}\n  phases: {type: concurrency, requests: 2, concurrency: 1}\n",
            ),
            Some("/tmp/x".into()),
        )
        .expect("valid public config resolves");
        let v = serde_json::to_value(&run).unwrap();
        assert_eq!(
            v["cfg"]["datasets"][0]["prompts"]["corpus"],
            serde_json::json!("coding")
        );
    }

    #[test]
    fn runtime_dispatch_is_yaml_authorable() {
        let run = resolve_str(
            &cfg(
                "  dataset: {prompts: {isl: 128}}\n  phases: {type: concurrency, requests: 2, concurrency: 1}\n  runtime: {dispatch: sharded}\n",
            ),
            Some("/tmp/x".into()),
        )
        .expect("valid config resolves");
        assert_eq!(
            run.cfg.runtime.expect("runtime present").dispatch,
            Some(aiperf_runtime::engine::protocol::DispatchMode::Sharded)
        );
    }

    #[test]
    fn runtime_dispatch_absent_defaults_to_global_none() {
        let run = resolve_str(
            &cfg(
                "  dataset: {prompts: {isl: 128}}\n  phases: {type: concurrency, requests: 2, concurrency: 1}\n",
            ),
            Some("/tmp/x".into()),
        )
        .expect("valid config resolves");
        // No `runtime.dispatch` authored: the typed field stays `None`, which the
        // protocol-v2 wire decode resolves to `DispatchMode::Global`.
        assert_eq!(run.cfg.runtime.expect("runtime present").dispatch, None);
    }

    #[test]
    fn runtime_dispatch_rejects_unknown_value() {
        let e = err(
            "  dataset: {prompts: {isl: 128}}\n  phases: {type: concurrency, requests: 1, concurrency: 1}\n  runtime: {dispatch: bogus}\n",
        );
        assert!(e.contains("dispatch") || e.contains("bogus"), "{e}");
    }

    #[test]
    fn endpoint_control_hooks_accept_bool_or_object_yaml() {
        let run = resolve_str(
            r#"
schemaVersion: "2.0"
benchmark:
  model: m
  endpoint:
    type: chat
    url: 127.0.0.1:8000
    reset_kv_cache: true
    server_profiler:
      timeout_seconds: 10
      start_path: /start_profile
      stop_path: /stop_profile
  dataset:
    prompts:
      isl: 128
  phases:
    type: concurrency
    requests: 2
    concurrency: 1
"#,
            Some("/tmp/x".into()),
        )
        .expect("valid config resolves");
        let endpoint = run.cfg.endpoint.expect("endpoint present");
        assert!(endpoint.reset_kv_cache.is_some());
        let server_profiler = endpoint.server_profiler.expect("server_profiler enabled");
        assert_eq!(server_profiler.timeout_seconds, Some(10.0));
        assert_eq!(
            server_profiler.start_path.as_deref(),
            Some("/start_profile")
        );
        assert_eq!(server_profiler.stop_path.as_deref(), Some("/stop_profile"));
    }

    /// Task 1 guard: a representative benchmark YAML resolves into a typed
    /// `BenchmarkConfig` carrying all four core sections (model, one dataset,
    /// one phase, endpoint). This locks in the YAML -> typed contract that the
    /// producer collapse must preserve byte-for-byte on the `--execute` wire.
    #[test]
    fn representative_yaml_populates_typed_benchmark_config() {
        let run = resolve_str(
            &cfg(
                "  dataset: {prompts: {isl: 128}}\n  phases: {type: concurrency, requests: 2, concurrency: 4}\n",
            ),
            Some("/tmp/x".into()),
        )
        .expect("valid config resolves");
        let c = &run.cfg;
        // Model selection is populated from the `model:` shorthand.
        let models = c.models.as_ref().expect("models present");
        let models_v = serde_json::to_value(models).unwrap();
        assert_eq!(
            models_v["items"][0]["name"],
            serde_json::json!("m"),
            "model name should survive into typed models: {models_v}"
        );
        // Exactly one canonical dataset, synthetic.
        let datasets = c.datasets.as_ref().expect("datasets present");
        assert_eq!(datasets.len(), 1, "one canonical dataset");
        let ds_v = serde_json::to_value(&datasets[0]).unwrap();
        assert_eq!(ds_v["type"], serde_json::json!("synthetic"));
        // Exactly one phase, concurrency-shaped.
        let phases = c.phases.as_ref().expect("phases present");
        assert_eq!(phases.len(), 1, "one phase");
        let ph_v = serde_json::to_value(&phases[0]).unwrap();
        assert_eq!(ph_v["concurrency"], serde_json::json!(4));
        // Endpoint profile is present with the authored type.
        let ep_v = serde_json::to_value(c.endpoint.as_ref().expect("endpoint present")).unwrap();
        assert_eq!(ep_v["type"], serde_json::json!("chat"));
    }

    /// Task 2 guard: an explicit CLI loadgen flag overrides the YAML-authored
    /// value, while an unset flag leaves the YAML value intact. This pins the
    /// flag-overlay-then-validate-once semantics the collapse must retain.
    #[test]
    fn cli_flag_overlay_wins_over_yaml_unset_flags_leave_yaml() {
        // `ProfileFlags` (clap derive) overflows the default test-thread stack;
        // run the parse+resolve on a generous worker stack (see flags.rs tests).
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let yaml = cfg(
                    "  dataset: {prompts: {isl: 128}}\n  phases: {type: concurrency, requests: 2, concurrency: 1}\n",
                );
                let raw: serde_json::Value = serde_yaml::from_str(&yaml).unwrap();
                let expanded = crate::expand::expand_config(raw).unwrap();
                let flags = crate::flags::ProfileFlags::parse_from_args(&[
                    "--concurrency".to_string(),
                    "8".to_string(),
                ])
                .expect("flags parse");
                let run = resolve_expanded_value(
                    expanded,
                    Some("/tmp/x".into()),
                    Some(&flags),
                )
                .expect("overlay resolves");
                let ph_v = serde_json::to_value(&run.cfg.phases.as_ref().unwrap()[0]).unwrap();
                // Overlaid `--concurrency 8` wins over the authored `concurrency: 1`.
                assert_eq!(
                    ph_v["concurrency"],
                    serde_json::json!(8),
                    "explicit --concurrency must override YAML"
                );
                // An unset flag (request rate) leaves the phase rate-free.
                assert!(
                    ph_v.get("request_rate").is_none()
                        || ph_v["request_rate"].is_null(),
                    "unset --request-rate must not inject a rate: {ph_v}"
                );
            })
            .expect("spawn worker")
            .join()
            .expect("worker panicked");
    }

    /// Increment A guard: an explicitly-set operational bool flag (`--streaming`)
    /// now overlays a YAML-authored endpoint, while an unset flag leaves the
    /// config value intact (byte-identical to no overlay).
    #[test]
    fn explicit_bool_flag_overlays_yaml_unset_leaves_config() {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                let yaml = cfg(
                    "  dataset: {prompts: {isl: 128}}\n  phases: {type: concurrency, requests: 2, concurrency: 1}\n",
                );
                let raw: serde_json::Value = serde_yaml::from_str(&yaml).unwrap();
                let expanded = crate::expand::expand_config(raw).unwrap();

                // No `--streaming` flag: the config default (streaming: false) stands.
                let unset = crate::flags::ProfileFlags::parse_from_args(&[
                    "--concurrency".to_string(),
                    "1".to_string(),
                ])
                .expect("flags parse");
                let run = resolve_expanded_value(
                    expanded.clone(),
                    Some("/tmp/x".into()),
                    Some(&unset),
                )
                .expect("overlay resolves");
                let v = serde_json::to_value(&run).unwrap();
                assert_eq!(
                    v["cfg"]["endpoint"]["streaming"],
                    serde_json::json!(false),
                    "unset --streaming must not flip the config default"
                );

                // Explicit `--streaming`: overlays the config-derived endpoint.
                let set = crate::flags::ProfileFlags::parse_from_args(&[
                    "--streaming".to_string(),
                ])
                .expect("flags parse");
                let run = resolve_expanded_value(
                    expanded,
                    Some("/tmp/x".into()),
                    Some(&set),
                )
                .expect("overlay resolves");
                let v = serde_json::to_value(&run).unwrap();
                assert_eq!(
                    v["cfg"]["endpoint"]["streaming"],
                    serde_json::json!(true),
                    "explicit --streaming must overlay the YAML endpoint"
                );
            })
            .expect("spawn worker")
            .join()
            .expect("worker panicked");
    }
}
