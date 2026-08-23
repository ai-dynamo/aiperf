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
use crate::model::dataset::{Distribution, RecordedAgentGraphConfig};
use crate::model::endpoint::{ResetKvCacheConfig, ServerProfilerConfig};
use crate::model::transport::{DynosimConfig, Transport, WebSocketTransportConfig};

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
    warn_unimplemented_keys(&file);
    // A nested runtime block takes precedence over the top-level block.
    if file.benchmark.runtime.is_none() {
        file.benchmark.runtime = file.runtime.take();
    }
    let mut inputs = file.benchmark.into_inputs(artifact_dir, random_seed)?;
    apply_cli_overrides(&mut inputs, overrides)?;
    Ok(inputs)
}

/// Config keys the native loader parses for Python-CLI compatibility but does
/// not act on. Each is accepted so an existing config keeps loading, and each is
/// warned about so a run never silently ignores an authored intent.
///
/// This is the config-surface twin of `profile::UNIMPLEMENTED_FLAGS`: entries
/// leave the table by gaining a consumer, not by being deleted, since deleting
/// one returns the key to silently-ignored — the failure this guards.
const UNIMPLEMENTED_KEYS: &[(&str, fn(&ConfigFile) -> bool)] = &[
    ("plot", |c| c.plot.is_some()),
    ("runtime.ui", |c| {
        c.benchmark
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.ui.as_ref())
            .is_some()
            || c.runtime
                .as_ref()
                .and_then(|runtime| runtime.ui.as_ref())
                .is_some()
    }),
];

/// Warn once for every authored config key the native loader does not act on.
fn warn_unimplemented_keys(file: &ConfigFile) {
    let authored: Vec<&str> = UNIMPLEMENTED_KEYS
        .iter()
        .filter(|(_, is_set)| is_set(file))
        .map(|(name, _)| *name)
        .collect();
    if !authored.is_empty() {
        tracing::warn!(
            keys = authored.join(", "),
            "ignored by the native runtime; these config keys are accepted for \
             compatibility with the Python CLI and have no effect on this run"
        );
    }
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
        // If synthetic entries fell back to the default (no explicit num_conversations,
        // dataset.entries, or phase.requests in YAML), use the CLI --request-count as
        // the entry pool size. Matches Python _resolve_entries fallback behavior.
        if inputs.entries == load::DEFAULT_ENTRIES {
            if let Ok(n) = u32::try_from(v) {
                inputs.entries = n;
            }
        }
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

/// The envelope `multiRun:` block. Each field is the config spelling of an
/// existing `--num-profile-runs` / `--profile-run-cooldown-seconds` /
/// `--confidence-level` / seed / warmup flag; the block only selects trial
/// repetition policy, so it is projected onto the flags rather than onto
/// `Inputs`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MultiRunSection {
    /// Trials per variation (`--num-profile-runs`).
    #[serde(default, alias = "numRuns")]
    num_runs: Option<u32>,
    /// Inter-trial cooldown (`--profile-run-cooldown-seconds`).
    #[serde(default, alias = "cooldownSeconds")]
    cooldown_seconds: Option<f64>,
    /// Confidence level for the aggregate (`--confidence-level`).
    #[serde(default, alias = "confidenceLevel")]
    confidence_level: Option<f64>,
    /// Reuse one seed across trials (`--set-consistent-seed`).
    #[serde(default, alias = "setConsistentSeed")]
    set_consistent_seed: Option<bool>,
    /// Skip warmup on trials past the first
    /// (`--profile-run-disable-warmup-after-first`).
    #[serde(default, alias = "disableWarmupAfterFirst")]
    disable_warmup_after_first: Option<bool>,
}

/// Project an authored envelope `multiRun:` block onto `flags`, letting an
/// explicit command-line flag win over the config. Applied before multi-run
/// validation so an out-of-range `numRuns` fails with the same message the flag
/// produces.
pub fn apply_multi_run(
    base: &serde_json::Value,
    flags: &mut crate::flags::ProfileFlags,
) -> anyhow::Result<()> {
    let Some(raw) = base.get("multiRun").or_else(|| base.get("multi_run")) else {
        return Ok(());
    };
    let section: MultiRunSection =
        serde_json::from_value(raw.clone()).map_err(|e| anyhow::anyhow!("multiRun: {e}"))?;
    // A flag authored on the command line is the more specific intent and wins;
    // otherwise the config value takes effect.
    if flags.num_profile_runs.is_none() {
        flags.num_profile_runs = section.num_runs;
    }
    if flags.profile_run_cooldown_seconds.is_none() {
        flags.profile_run_cooldown_seconds = section.cooldown_seconds;
    }
    if flags.confidence_level.is_none() {
        flags.confidence_level = section.confidence_level;
    }
    if flags.set_consistent_seed.is_none() && flags.no_set_consistent_seed.is_none() {
        flags.set_consistent_seed = section.set_consistent_seed;
    }
    if flags.profile_run_disable_warmup_after_first.is_none()
        && flags.no_profile_run_disable_warmup_after_first.is_none()
    {
        flags.profile_run_disable_warmup_after_first = section.disable_warmup_after_first;
    }
    Ok(())
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
#[serde(deny_unknown_fields)]
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
    /// Weighted mixture components; selects a multi-modal distribution.
    peaks: Option<Vec<PeakFields>>,
}

/// One mixture component, authored inline as `{mean: 128, stddev: 20, weight: 60}`.
/// The wire form nests the component under `distribution:`, so [`dist_from`]
/// canonicalizes; `weight` defaults to an equal share.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PeakFields {
    value: Option<f64>,
    mean: Option<f64>,
    stddev: Option<f64>,
    median: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
    #[serde(default = "one")]
    weight: f64,
}

/// Default mixture weight, giving every peak an equal share when `weight:` is
/// omitted from all of them.
fn one() -> f64 {
    1.0
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
#[serde(deny_unknown_fields)]
struct ConfigFile {
    /// Config-schema version (`schemaVersion: "2.0"`). Accepted and recorded so
    /// authoring it is not an unknown key; the loader targets one schema.
    #[serde(default, alias = "schemaVersion")]
    #[allow(dead_code)]
    schema_version: Option<String>,
    /// `sweep:` block, consumed by [`crate::sweep::yaml_sweep::parse`] before this
    /// struct sees the value and stripped per variation. Declared so the key is
    /// legal on the single-run path rather than rejected as unknown.
    #[serde(default)]
    #[allow(dead_code)]
    sweep: Option<serde_json::Value>,
    /// `variables:` block, consumed by [`crate::expand`] during Jinja expansion.
    /// Declared for the same reason as `sweep`.
    #[serde(default)]
    #[allow(dead_code)]
    variables: Option<serde_json::Value>,
    /// `multiRun:` block, read by [`apply_multi_run`] before this struct is
    /// deserialized. Declared so the key is legal rather than rejected as unknown.
    #[serde(default, alias = "multiRun")]
    #[allow(dead_code)]
    multi_run: Option<serde_json::Value>,
    /// `plot:` visualization envelope. The native binary has no plotting
    /// command, so this is accepted for compatibility with the Python CLI and
    /// warned about by [`warn_unimplemented_keys`] rather than silently dropped.
    #[serde(default)]
    plot: Option<serde_json::Value>,
    benchmark: Benchmark,
    /// Top-level deterministic run seed (`randomSeed`).
    #[serde(default, alias = "randomSeed")]
    random_seed: Option<u64>,
    /// Top-level worker/cell runtime policy (`runtime.cells`).
    #[serde(default)]
    runtime: Option<RuntimeSection>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Endpoint and environment provenance for recorded-agent replay.
    metadata: Option<MetadataSection>,
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
#[serde(deny_unknown_fields)]
struct MetadataSection {
    /// Free-form endpoint hardware description.
    hardware: Option<String>,
    /// Endpoint placement relative to recorded-agent tool execution.
    #[serde(default, alias = "endpointPlacement")]
    endpoint_placement: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactsSection {
    /// Run artifact directory (the `--artifact-dir` flag overrides it).
    dir: Option<String>,
    /// Timeslice window, seconds (wire `metrics.slice_duration_seconds`).
    #[serde(default, alias = "sliceDuration", deserialize_with = "de_duration_opt")]
    slice_duration: Option<f64>,
    /// Per-record export formats (`[jsonl,csv,parquet]`) or `false` to disable.
    records: Option<RecordsFormats>,
    /// Summary export formats (`[json,csv]`). Unauthored ships both.
    #[serde(default)]
    summary: Option<Vec<String>>,
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
    /// Files materialized into the run directory before the benchmark starts.
    #[serde(default, alias = "userFiles")]
    user_files: Option<Vec<UserFileSection>>,
}

/// One `artifacts.userFiles` entry. `format` is inferred from `content` when
/// omitted: a string is `text`, structured content is `json`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct UserFileSection {
    /// POSIX-style path relative to the run directory.
    path: String,
    /// `json`, `yaml`, or `text`.
    #[serde(default)]
    format: Option<String>,
    /// Structured content (serialized per `format`) or a text body.
    content: serde_json::Value,
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
#[serde(deny_unknown_fields)]
struct GpuTelemetrySection {
    enabled: Option<bool>,
    urls: Option<Vec<String>>,
    /// Collector backend: the DCGM exporter scraped over HTTP, or one of the two
    /// local collectors (`pynvml`, `amdsmi`) that read the host's own driver.
    /// Anything else is rejected rather than downgraded to DCGM.
    #[serde(default)]
    collector: Option<String>,
    /// Display mode. `summary` is what the native runtime does; the live TUI
    /// (`realtime_dashboard`) has no native implementation, so it is rejected
    /// rather than accepted and silently rendered as a summary table.
    #[serde(default)]
    mode: Option<String>,
    /// Custom DCGM exporter field definitions (CSV). DCGM-only.
    #[serde(default, alias = "metricsFile")]
    metrics_file: Option<String>,
}

impl GpuTelemetrySection {
    /// Reject authored values whose behavior the native runtime does not
    /// implement, so an unsupported backend or display mode fails loudly instead
    /// of resolving to the one thing the runtime always does.
    fn validate(&self) -> anyhow::Result<()> {
        let collector = self.collector.as_deref().unwrap_or("dcgm");
        if !matches!(collector, "dcgm" | "pynvml" | "amdsmi") {
            anyhow::bail!(
                "gpuTelemetry.collector {collector:?} is not supported \
                 (the native runtime implements \"dcgm\", \"pynvml\", and \"amdsmi\")"
            );
        }
        if let Some(mode) = self.mode.as_deref()
            && mode != "summary"
        {
            anyhow::bail!(
                "gpuTelemetry.mode {mode:?} is not supported \
                 (the native runtime implements only \"summary\")"
            );
        }
        // The local collectors read the host driver in-process: neither a scrape
        // endpoint nor a DCGM exporter field CSV applies to them, so authoring
        // one fails here instead of being accepted and dropped at lowering.
        if collector != "dcgm" {
            if self.urls.as_ref().is_some_and(|urls| !urls.is_empty()) {
                anyhow::bail!(
                    "gpuTelemetry.urls is not supported by the {collector:?} collector \
                     (it reads the local host, not a scrape endpoint)"
                );
            }
            if self.metrics_file.is_some() {
                anyhow::bail!(
                    "gpuTelemetry.metricsFile is not supported by the {collector:?} collector \
                     (custom field definitions apply to the DCGM exporter only)"
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServerMetricsSection {
    enabled: Option<bool>,
    urls: Option<Vec<String>>,
    formats: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NetworkLatencySection {
    #[serde(default)]
    enabled: bool,
    #[serde(default, alias = "meanMs")]
    mean_ms: Option<f64>,
    #[serde(default, alias = "pingInterval")]
    ping_interval: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OtelSection {
    #[serde(alias = "metricsUrl")]
    metrics_url: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MlflowSection {
    #[serde(default, alias = "trackingUri")]
    tracking_uri: Option<String>,
    experiment: Option<String>,
    #[serde(default, alias = "runName")]
    run_name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WandbSection {
    project: Option<String>,
    entity: Option<String>,
    #[serde(default, alias = "runName")]
    run_name: Option<String>,
    #[serde(default, alias = "syncUrl")]
    sync_url: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeSection {
    // Compatibility-only UI selection. The native runtime has no UI renderer.
    ui: Option<String>,
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
    /// Transport-specific fields stay flat beside the discriminator and are
    /// decoded only after `type` selects their public DTO.
    #[serde(flatten)]
    options: serde_json::Map<String, serde_json::Value>,
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
#[serde(deny_unknown_fields)]
struct EndpointSection {
    #[serde(rename = "type")]
    endpoint_type: Option<String>,
    /// Endpoint URL(s); `url:` (single/list) or plural `urls:`. Optional:
    /// DynoSim endpoints carry no URL (the sentinel is injected).
    #[serde(alias = "urls")]
    url: Option<StringOrVec>,
    /// Multi-URL selection strategy. `round_robin` is the only strategy the
    /// native runtime implements (`ancillary::url_selector` always builds
    /// `RoundRobinUrlSelector`), so it is honored by construction and any other
    /// name is rejected during `into_inputs` rather than accepted and silently
    /// downgraded to round-robin.
    #[serde(default, alias = "urlStrategy")]
    url_strategy: Option<String>,
    #[serde(default)]
    streaming: bool,
    #[serde(default, alias = "apiKey")]
    api_key: Option<String>,
    /// Request timeout, seconds (wire `timeout_seconds`).
    timeout: Option<f64>,
    #[serde(default, alias = "connectionReuse")]
    connection_reuse: Option<String>,
    #[serde(default, alias = "sslVerify")]
    ssl_verify: Option<bool>,
    #[serde(default, alias = "udsPath")]
    uds_path: Option<String>,
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
#[serde(deny_unknown_fields)]
struct DatasetSection {
    /// Entry identity, consumed by the `sweep:` path resolver
    /// (`sweep::yaml_sweep::find_named_index`) so a dotted parameter like
    /// `datasets.main.format` targets this entry. Resolution happens on the raw
    /// config before this struct is deserialized, so the field is declared here
    /// only to keep the key legal rather than rejected as unknown.
    #[serde(default)]
    #[allow(dead_code)]
    name: Option<String>,
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
    /// Recorded-graph / trace synthesis transforms (`TraceSynthesisSpec`).
    synthesis: Option<SynthesisSection>,
    /// Recorded-agent replay settings (`dataset.graph`).
    graph: Option<RecordedAgentGraphConfig>,
    /// Synthetic image generation (`synthetic.images`).
    images: Option<ImageSection>,
    /// Synthetic audio generation (`synthetic.audio`).
    audio: Option<AudioSection>,
    /// Synthetic video generation (`synthetic.video`).
    video: Option<VideoSection>,
}

/// Authored `dataset.synthesis:` block. Mirrors the wire `TraceSynthesisSpec`,
/// whose multiplier fields are non-optional, so [`SynthesisSection::to_value`]
/// stamps the same identity defaults the `--synthesis-*` flag path uses.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SynthesisSection {
    corpus: Option<String>,
    #[serde(default, alias = "speedupRatio")]
    speedup_ratio: Option<f64>,
    #[serde(default, alias = "prefixLenMultiplier")]
    prefix_len_multiplier: Option<f64>,
    #[serde(default, alias = "prefixRootMultiplier")]
    prefix_root_multiplier: Option<u64>,
    #[serde(default, alias = "promptLenMultiplier")]
    prompt_len_multiplier: Option<f64>,
    #[serde(default, alias = "outputLenMultiplier")]
    output_len_multiplier: Option<f64>,
    #[serde(default, alias = "maxIsl")]
    max_isl: Option<u64>,
    #[serde(default, alias = "maxOsl")]
    max_osl: Option<u32>,
    #[serde(default, alias = "maxContextLength")]
    max_context_length: Option<u64>,
    #[serde(default, alias = "allowDatasetWrap")]
    allow_dataset_wrap: Option<bool>,
    #[serde(default, alias = "idleGapCapSeconds")]
    idle_gap_cap_seconds: Option<f64>,
    #[serde(default, alias = "trajectoryStartMinRatio")]
    trajectory_start_min_ratio: Option<f64>,
    #[serde(default, alias = "trajectoryStartMaxRatio")]
    trajectory_start_max_ratio: Option<f64>,
    #[serde(default, alias = "tStarRandomSeed")]
    t_star_random_seed: Option<u64>,
    #[serde(default, alias = "datasetSamplingStrategy")]
    dataset_sampling_strategy: Option<String>,
    #[serde(default, alias = "cacheBustTarget")]
    cache_bust_target: Option<String>,
}

impl SynthesisSection {
    /// Lower to the JSON object `Inputs::synthesis` carries to the dataset wire.
    fn to_value(&self) -> anyhow::Result<serde_json::Value> {
        // JSON has no non-finite numbers; reject rather than panic in `from_f64`.
        let f = |v: f64| -> anyhow::Result<serde_json::Value> {
            serde_json::Number::from_f64(v)
                .map(serde_json::Value::Number)
                .ok_or_else(|| {
                    anyhow::anyhow!("dataset.synthesis numeric value must be finite, got {v}")
                })
        };
        let mut m = serde_json::Map::new();
        m.insert(
            "speedup_ratio".into(),
            f(self.speedup_ratio.unwrap_or(1.0))?,
        );
        m.insert(
            "prefix_len_multiplier".into(),
            f(self.prefix_len_multiplier.unwrap_or(1.0))?,
        );
        m.insert(
            "prefix_root_multiplier".into(),
            serde_json::Value::from(self.prefix_root_multiplier.unwrap_or(1)),
        );
        m.insert(
            "prompt_len_multiplier".into(),
            f(self.prompt_len_multiplier.unwrap_or(1.0))?,
        );
        m.insert(
            "output_len_multiplier".into(),
            f(self.output_len_multiplier.unwrap_or(1.0))?,
        );
        m.insert(
            "idle_gap_cap_seconds".into(),
            f(self.idle_gap_cap_seconds.unwrap_or(60.0))?,
        );
        if let Some(v) = &self.corpus {
            m.insert("corpus".into(), serde_json::Value::String(v.clone()));
        }
        if let Some(v) = self.max_isl {
            m.insert("max_isl".into(), serde_json::Value::from(v));
        }
        if let Some(v) = self.max_osl {
            m.insert("max_osl".into(), serde_json::Value::from(v));
        }
        if let Some(v) = self.max_context_length {
            m.insert("max_context_length".into(), serde_json::Value::from(v));
        }
        if let Some(v) = self.allow_dataset_wrap {
            m.insert("allow_dataset_wrap".into(), serde_json::Value::Bool(v));
        }
        if let Some(v) = self.trajectory_start_min_ratio {
            m.insert("trajectory_start_min_ratio".into(), f(v)?);
        }
        if let Some(v) = self.trajectory_start_max_ratio {
            m.insert("trajectory_start_max_ratio".into(), f(v)?);
        }
        if let Some(v) = self.t_star_random_seed {
            m.insert("t_star_random_seed".into(), serde_json::Value::from(v));
        }
        if let Some(v) = &self.dataset_sampling_strategy {
            m.insert(
                "dataset_sampling_strategy".into(),
                serde_json::Value::String(v.clone()),
            );
        }
        if let Some(v) = &self.cache_bust_target {
            m.insert(
                "cache_bust_target".into(),
                serde_json::Value::String(v.clone()),
            );
        }
        Ok(serde_json::Value::Object(m))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
struct VideoAudioSection {
    channels: Option<u32>,
    codec: Option<String>,
    depth: Option<u32>,
    #[serde(default, alias = "sampleRate")]
    sample_rate: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Weighted `(isl, osl)` mixture, the YAML form of `--seq-dist`. Overrides
    /// the scalar `isl`/`osl` above when authored.
    #[serde(default, alias = "sequenceDistribution")]
    sequence_distribution: Option<Vec<SeqDistEntrySection>>,
}

/// One `sequenceDistribution:` entry: a weighted `(isl, osl)` pair.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeqDistEntrySection {
    isl: NumOrDist,
    osl: NumOrDist,
    probability: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    /// Exclude this phase's records from the reported results. Defaults to the
    /// phase role (warmup excluded, profiling included) when unauthored.
    #[serde(default, alias = "excludeFromResults")]
    exclude_from_results: Option<bool>,
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
#[serde(deny_unknown_fields)]
struct CancellationSection {
    rate: f64,
    delay: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
        // Honored by construction (round-robin is the only native selector), so
        // reject any other name instead of running a strategy nobody asked for.
        if let Some(strategy) = self.endpoint.url_strategy.as_deref()
            && strategy != "round_robin"
        {
            anyhow::bail!(
                "endpoint.urlStrategy {strategy:?} is not supported \
                 (the native runtime implements only \"round_robin\")"
            );
        }

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
        let sequence_distribution = dataset
            .as_ref()
            .and_then(|d| d.prompts.as_ref())
            .and_then(|p| p.sequence_distribution.as_ref())
            .map(|entries| {
                entries
                    .iter()
                    .map(|e| crate::model::dataset::SeqDistEntry {
                        isl: clone_num_or_dist(&e.isl),
                        osl: clone_num_or_dist(&e.osl),
                        probability: e.probability,
                    })
                    .collect()
            });
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
        let synthesis = match dataset.as_ref().and_then(|d| d.synthesis.as_ref()) {
            Some(section) => Some(section.to_value()?),
            None => None,
        };

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
        let recorded_agent_graph = dataset.as_ref().and_then(|dataset| dataset.graph.clone());
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

        // Resolution order (matches Python _resolve_entries): explicit num_conversations,
        // explicit dataset.entries, then fallback to phase.requests so a single
        // `requests: N` / `--request-count N` invocation produces N unique entries.
        let entries = num_conversations
            .or(dataset_entries)
            .or_else(|| phase.requests.and_then(|n| u32::try_from(n).ok()))
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
        if let Some(gpu) = self.gpu_telemetry.as_ref() {
            gpu.validate()?;
        }
        let (gpu_enabled, gpu_urls, gpu_collector, gpu_metrics_file) = self
            .gpu_telemetry
            .as_ref()
            .map(|g| {
                (
                    g.enabled.unwrap_or(true),
                    g.urls.clone().unwrap_or_default(),
                    g.collector.clone(),
                    g.metrics_file.clone(),
                )
            })
            .unwrap_or((true, Vec::new(), None, None));

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
                sync_url: w.sync_url.clone(),
            })
            .unwrap_or(crate::model::export::WandbParams {
                project: None,
                entity: None,
                run_name: None,
                tags: Vec::new(),
                sync_url: None,
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
        // `artifacts.summary`: an authored list narrows the summary artifacts, so
        // an unknown format name must fail rather than silently ship both.
        let summary_formats = match self.artifacts.as_ref().and_then(|a| a.summary.as_ref()) {
            Some(v) => {
                for f in v {
                    anyhow::ensure!(
                        f == "json" || f == "csv",
                        "artifacts.summary: unknown format {f:?} (expected `json` or `csv`)"
                    );
                }
                anyhow::ensure!(
                    !v.is_empty(),
                    "artifacts.summary: empty list; omit the key to keep both formats"
                );
                v.clone()
            }
            None => Vec::new(),
        };
        // `artifacts.userFiles`: rendered once here, so the runner materializes
        // bytes rather than re-rendering templates at run time.
        let mut user_files = Vec::new();
        for f in self
            .artifacts
            .as_ref()
            .and_then(|a| a.user_files.as_ref())
            .into_iter()
            .flatten()
        {
            let is_text = f.content.is_string();
            let format = match f.format.as_deref() {
                Some(v @ ("json" | "yaml" | "text")) => v,
                Some(other) => anyhow::bail!(
                    "artifacts.userFiles[{}]: unknown format {other:?} (expected `json`, `yaml`, or `text`)",
                    f.path
                ),
                // Inferred: a string body is text, structured content is JSON.
                None if is_text => "text",
                None => "json",
            };
            let content = match format {
                "text" => f.content.as_str().map(str::to_owned).ok_or_else(|| {
                    anyhow::anyhow!(
                        "artifacts.userFiles[{}]: format `text` requires a string `content`",
                        f.path
                    )
                })?,
                "yaml" => serde_yaml::to_string(&f.content)?,
                _ => serde_json::to_string_pretty(&f.content)?,
            };
            user_files.push(crate::model::artifacts::UserFile {
                path: f.path.clone(),
                format: format.to_owned(),
                content,
            });
        }

        let export_raw = self.artifacts.as_ref().is_some_and(|a| a.raw);
        let show_trace_timing = self.artifacts.as_ref().is_some_and(|a| a.show_trace_timing);
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
        let hardware_description = self
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.hardware.clone());
        let endpoint_placement = self
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.endpoint_placement.clone())
            .unwrap_or_else(|| "unknown".to_string());

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
            ssl_verify: self.endpoint.ssl_verify,
            uds_path: self.endpoint.uds_path.clone(),
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
            gpu_telemetry_collector: gpu_collector,
            gpu_telemetry_urls: gpu_urls,
            gpu_telemetry_metrics_file: gpu_metrics_file,
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
            summary_formats,
            user_files,
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
            sequence_distribution,
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
            recorded_agent_graph,
            hardware_description,
            endpoint_placement,
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
            random_pool_image_batch_size: None,
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
            synthesis,
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
    let reject_websocket_options = || -> anyhow::Result<()> {
        for field in [
            "fallback",
            "ping_interval_seconds",
            "stream_idle_timeout_seconds",
            "max_queued_commands",
            "max_queued_bytes",
            "max_frame_bytes",
            "max_message_bytes",
            "max_response_bytes",
        ] {
            anyhow::ensure!(
                !section.options.contains_key(field),
                "transport.type {:?} does not support WebSocket field {field:?}",
                section.transport_type
            );
        }
        Ok(())
    };
    let decode_options = || serde_json::Value::Object(section.options.clone());
    let dynosim = || -> anyhow::Result<DynosimConfig> {
        reject_websocket_options()?;
        let mut cfg: DynosimConfig = serde_json::from_value(decode_options()).map_err(|error| {
            anyhow::anyhow!("transport.type {:?}: {error}", section.transport_type)
        })?;
        cfg.normalize();
        Ok(cfg)
    };
    match section.transport_type.as_str() {
        "http" => {
            reject_websocket_options()?;
            Ok(Transport::Http)
        }
        "grpc" => {
            reject_websocket_options()?;
            Ok(Transport::Grpc)
        }
        "dynosim_offline" => Ok(Transport::DynosimOffline(dynosim()?)),
        "dynosim_online" => Ok(Transport::DynosimOnline(dynosim()?)),
        "dry_run" => {
            reject_websocket_options()?;
            let config = serde_json::from_value(decode_options()).map_err(|error| {
                anyhow::anyhow!("transport.type {:?}: {error}", section.transport_type)
            })?;
            Ok(Transport::DryRun(config))
        }
        "websocket" => serde_json::from_value::<WebSocketTransportConfig>(decode_options())
            .map(Transport::Websocket)
            .map_err(|error| {
                anyhow::anyhow!("transport.type {:?}: {error}", section.transport_type)
            }),
        other => anyhow::bail!("unknown transport.type {other:?}"),
    }
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
            exclude_from_results: section
                .exclude_from_results
                .unwrap_or(role == PhaseRole::Warmup),
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
        peaks: d.peaks.as_ref().map(|peaks| {
            peaks
                .iter()
                .map(|p| crate::model::dataset::Peak {
                    distribution: normalize_dist(Distribution {
                        value: p.value,
                        mean: p.mean,
                        stddev: p.stddev,
                        median: p.median,
                        min: p.min,
                        max: p.max,
                        ..Default::default()
                    }),
                    weight: p.weight,
                })
                .collect()
        }),
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
    use super::{ConfigFile, UNIMPLEMENTED_KEYS, resolve_expanded_value, resolve_str};

    #[test]
    fn websocket_transport_maps_every_authored_value() {
        let run = resolve_str(
            &cfg("  transport:\n\
                 \x20   type: websocket\n\
                 \x20   fallback: http_sse\n\
                 \x20   ping_interval_seconds: 12.5\n\
                 \x20   stream_idle_timeout_seconds: 34.5\n\
                 \x20   max_queued_commands: 7\n\
                 \x20   max_queued_bytes: 4096\n\
                 \x20   max_frame_bytes: 1024\n\
                 \x20   max_message_bytes: 2048\n\
                 \x20   max_response_bytes: 8192\n\
                 \x20 phases: {type: concurrency, requests: 1, concurrency: 1}\n"),
            Some("/tmp/x".into()),
        )
        .expect("websocket YAML resolves");
        let transport = run.cfg.transport.as_ref().expect("resolved transport");

        assert_eq!(
            serde_json::to_value(transport).expect("transport serializes"),
            serde_json::json!({
                "type": "websocket",
                "fallback": "http_sse",
                "ping_interval_seconds": 12.5,
                "stream_idle_timeout_seconds": 34.5,
                "max_queued_commands": 7,
                "max_queued_bytes": 4096,
                "max_frame_bytes": 1024,
                "max_message_bytes": 2048,
                "max_response_bytes": 8192,
            })
        );
    }

    #[test]
    fn websocket_transport_rejects_unknown_yaml_field() {
        for (field, value) in [
            ("surprise", "true"),
            ("engine_profile", "profile"),
            ("ttft_ms", "1"),
        ] {
            let body = format!(
                "  transport:\n    type: websocket\n    {field}: {value}\n  phases: {{type: concurrency, requests: 1, concurrency: 1}}\n"
            );
            let error = err(&body);
            assert!(
                error.contains("unknown field") && error.contains(field),
                "{error}"
            );
        }
    }

    #[test]
    fn other_transports_reject_websocket_only_yaml_fields() {
        let transports = [
            "http",
            "grpc",
            "dry_run",
            "dynosim_offline",
            "dynosim_online",
        ];
        let websocket_fields = [
            ("fallback", "http_sse"),
            ("ping_interval_seconds", "1"),
            ("stream_idle_timeout_seconds", "1"),
            ("max_queued_commands", "1"),
            ("max_queued_bytes", "1"),
            ("max_frame_bytes", "1"),
            ("max_message_bytes", "1"),
            ("max_response_bytes", "1"),
        ];

        for transport in transports {
            for (field, value) in websocket_fields {
                let body = format!(
                    "  transport:\n    type: {transport}\n    {field}: {value}\n  phases: {{type: concurrency, requests: 1, concurrency: 1}}\n"
                );
                let error = err(&body);
                assert!(
                    error.contains(field),
                    "transport {transport} must reject {field}: {error}"
                );
            }
        }
    }

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

    /// The authored inline peak form must reach the wire as the nested
    /// `{distribution, weight}` shape the runtime's multimodal sniff expects;
    /// parsing it without lowering would run a default single-mode ISL.
    #[test]
    fn inline_peaks_lower_to_the_wire_mixture() {
        let run = resolve_str(
            &cfg("  dataset:\n    type: synthetic\n    prompts:\n      isl:\n        peaks:\n          - {mean: 128, stddev: 20, weight: 60}\n          - {mean: 2048, median: 1891, weight: 40}\n  phases: {type: concurrency, requests: 1, concurrency: 1}\n"),
            Some("/tmp/x".into()),
        )
        .expect("inline peaks resolve");
        let dataset = run
            .cfg
            .datasets
            .as_ref()
            .and_then(|d| d.first())
            .expect("dataset");
        let crate::model::dataset::Dataset::Synthetic(synthetic) = dataset else {
            panic!("expected a synthetic dataset");
        };
        let isl = &synthetic.prompts.isl;
        let peaks = isl.peaks.as_ref().expect("peaks survive resolution");
        assert_eq!(peaks.len(), 2);
        assert_eq!(peaks[0].weight, 60.0);
        assert_eq!(peaks[0].distribution.mean, Some(128.0));
        assert_eq!(peaks[0].distribution.stddev, Some(20.0));
        // The log-normal peak keeps median and must not gain a stddev default.
        assert_eq!(peaks[1].distribution.median, Some(1891.0));
        assert_eq!(peaks[1].distribution.stddev, None);
        // A peak-bearing parent is not itself a normal distribution.
        assert_eq!(isl.stddev, None);
        assert_eq!(isl.mean, None);
    }

    /// An authored `dataset.synthesis:` block must reach the wire, not merely
    /// parse: the multipliers drive the recorded-graph transforms, so dropping
    /// them silently replays the trace at its recorded scale.
    #[test]
    fn authored_synthesis_reaches_the_wire() {
        let run = resolve_str(
            &cfg("  dataset:\n    type: file\n    path: /tmp/trace.jsonl\n    format: mooncake_trace\n    synthesis:\n      speedupRatio: 2.0\n      promptLenMultiplier: 1.5\n      maxIsl: 8192\n      maxOsl: 2048\n  phases: {type: concurrency, requests: 1, concurrency: 1}\n"),
            Some("/tmp/x".into()),
        )
        .expect("authored synthesis resolves");
        let dataset = run
            .cfg
            .datasets
            .as_ref()
            .and_then(|d| d.first())
            .expect("dataset");
        let crate::model::dataset::Dataset::File(file) = dataset else {
            panic!("expected a file dataset");
        };
        let synthesis = file.synthesis.as_ref().expect("synthesis survives");
        assert_eq!(synthesis["speedup_ratio"], 2.0);
        assert_eq!(synthesis["prompt_len_multiplier"], 1.5);
        assert_eq!(synthesis["max_isl"], 8192);
        assert_eq!(synthesis["max_osl"], 2048);
        // Omitted multipliers take the wire's non-optional identity defaults.
        assert_eq!(synthesis["prefix_len_multiplier"], 1.0);
        assert_eq!(synthesis["output_len_multiplier"], 1.0);
        assert_eq!(synthesis["prefix_root_multiplier"], 1);
        // Unauthored optional keys stay absent rather than materializing nulls.
        assert!(synthesis.get("cache_bust_target").is_none());
    }

    /// `round_robin` is the only native selector, so it must be accepted and
    /// every other name rejected — accepting one and silently round-robining
    /// would report a strategy the run never used.
    #[test]
    fn url_strategy_accepts_round_robin_and_rejects_others() {
        let with = |strategy: &str| {
            resolve_str(
                &cfg(&format!(
                    "  endpoint:\n    urls: [http://a:8000/v1, http://b:8000/v1]\n    urlStrategy: {strategy}\n  phases: {{type: concurrency, requests: 1, concurrency: 1}}\n"
                )),
                Some("/tmp/x".into()),
            )
        };
        with("round_robin").expect("round_robin is the native strategy");
        let err = with("random").expect_err("an unimplemented strategy must fail");
        assert!(
            err.to_string().contains("urlStrategy"),
            "error should name the key: {err}"
        );
    }

    /// `dcgm`, `pynvml`, and `amdsmi` are the three collectors the native runtime
    /// builds; anything else, and the `realtime_dashboard` mode that has no native
    /// renderer, must fail rather than resolve to DCGM-and-a-summary-table under
    /// another name.
    #[test]
    fn gpu_telemetry_rejects_backends_and_modes_it_cannot_honor() {
        let with = |body: &str| {
            resolve_str(
                &cfg(&format!(
                    "  gpuTelemetry:\n    enabled: true\n{body}  phases: {{type: concurrency, requests: 1, concurrency: 1}}\n"
                )),
                Some("/tmp/x".into()),
            )
        };
        with("    collector: dcgm\n    mode: summary\n").expect("the native backend and mode");
        let err = with("    collector: nvidia-smi\n").expect_err("no such native collector");
        assert!(
            err.to_string().contains("collector"),
            "error should name the key: {err}"
        );
        let err = with("    mode: realtime_dashboard\n").expect_err("no native TUI dashboard");
        assert!(
            err.to_string().contains("mode"),
            "error should name the key: {err}"
        );
    }

    /// The native local collectors read the host's own driver: they are
    /// authorable by name, and the DCGM-only scrape URLs and field CSV must fail
    /// rather than be accepted and silently dropped.
    #[test]
    fn gpu_telemetry_accepts_local_collectors_and_rejects_dcgm_only_options() {
        let with = |body: &str| {
            resolve_str(
                &cfg(&format!(
                    "  gpuTelemetry: {{{body}}}\n  phases: {{type: concurrency, requests: 1, concurrency: 1}}\n"
                )),
                Some("/tmp/x".into()),
            )
        };
        with("collector: pynvml, mode: summary").expect("the native NVML collector");
        with("collector: amdsmi").expect("the native AMD SMI collector");
        let err = with("collector: amdsmi, urls: [http://x]")
            .expect_err("a local collector scrapes no URL");
        assert!(
            err.to_string().contains("urls"),
            "should name the key: {err}"
        );
        let err = with("collector: pynvml, metricsFile: fields.csv")
            .expect_err("a local collector reads no DCGM field CSV");
        assert!(
            err.to_string().contains("metricsFile") || err.to_string().contains("metrics_file"),
            "should name the key: {err}"
        );
    }

    /// A `plot:` envelope must load and be reported as unacted-on: the native
    /// binary has no plotting command, so rejecting it would break working
    /// configs and accepting it silently would imply plots that never render.
    #[test]
    fn plot_envelope_loads_and_is_reported_as_unimplemented() {
        let text = cfg("  phases: {type: concurrency, requests: 1, concurrency: 1}\n")
            + "plot:\n  plot_config:\n    multi_run_plots: []\n";
        let expanded: serde_json::Value =
            serde_yaml::from_str(&text).expect("fixture is valid YAML");
        let file: ConfigFile =
            serde_json::from_value(expanded).expect("an authored plot envelope loads");
        assert!(
            UNIMPLEMENTED_KEYS
                .iter()
                .any(|(name, is_set)| *name == "plot" && is_set(&file)),
            "plot must be reported as accepted-but-unacted-on"
        );
    }

    /// `runtime.ui` remains part of the Config v2 compatibility surface, but the
    /// native runtime has no renderer. It must load and be reported as unacted-on,
    /// matching the accepted-but-unimplemented `--ui` flag alias.
    #[test]
    fn runtime_ui_loads_and_is_reported_as_unimplemented() {
        let text = cfg(
            "  dataset: {prompts: {isl: 128}}\n  phases: {type: concurrency, requests: 1, concurrency: 1}\n  runtime: {ui: none}\n",
        );
        let expanded: serde_json::Value =
            serde_yaml::from_str(&text).expect("fixture is valid YAML");
        let file: ConfigFile =
            serde_json::from_value(expanded).expect("an authored runtime UI setting loads");
        assert!(
            UNIMPLEMENTED_KEYS
                .iter()
                .any(|(name, is_set)| *name == "runtime.ui" && is_set(&file)),
            "runtime.ui must be reported as accepted-but-unacted-on"
        );
        resolve_str(&text, Some("/tmp/x".into())).expect("runtime.ui config resolves");
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

    /// When synthetic dataset has no explicit entries and no phase.requests,
    /// a CLI --request-count N should become the entry pool size (matching
    /// Python _resolve_entries fallback). When entries is set by YAML, CLI
    /// --request-count should not override it.
    #[test]
    fn cli_request_count_fills_synthetic_entries_when_defaulted() {
        use aiperf_runtime::config::model::dataset::Dataset;

        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                // --- Case 1: no entries, no phase.requests → CLI --request-count fills it ---
                let yaml = cfg("  dataset: {prompts: {isl: 128}}\n  phases: {type: concurrency, concurrency: 1}\n");
                let raw: serde_json::Value = serde_yaml::from_str(&yaml).unwrap();
                let expanded = crate::expand::expand_config(raw).unwrap();
                let flags = crate::flags::ProfileFlags::parse_from_args(&[
                    "--request-count".to_string(),
                    "500".to_string(),
                ])
                .expect("flags parse");
                let run = resolve_expanded_value(
                    expanded,
                    Some("/tmp/x".into()),
                    Some(&flags),
                )
                .expect("resolve");
                // Phase should have requests=500 from CLI override
                assert_eq!(
                    run.cfg.phases.as_ref().unwrap()[0].common.requests,
                    Some(500),
                    "CLI --request-count should set phase requests"
                );

                // --- Case 2: explicit YAML entries → CLI --request-count does NOT fill it ---
                let yaml_explicit = cfg(
                    "  dataset: {entries: 200, prompts: {isl: 128}}\n  \
                     phases: {type: concurrency, concurrency: 1}\n",
                );
                let raw: serde_json::Value = serde_yaml::from_str(&yaml_explicit).unwrap();
                let expanded = crate::expand::expand_config(raw).unwrap();
                let run = resolve_expanded_value(
                    expanded,
                    Some("/tmp/x".into()),
                    Some(&flags),
                )
                .expect("resolve with explicit entries");
                // Dataset should still have entries=200
                if let Dataset::Synthetic(syn) = &run.cfg.datasets.as_ref().unwrap()[0] {
                    assert_eq!(
                        syn.entries,
                        Some(200),
                        "explicit YAML entries must not be overridden by CLI --request-count"
                    );
                } else {
                    panic!("expected synthetic dataset");
                }

                // --- Case 3: YAML phase.requests fills entries (no CLI request-count) ---
                let yaml_requests = cfg(
                    "  dataset: {prompts: {isl: 128}}\n  \
                     phases: {type: concurrency, requests: 300, concurrency: 1}\n",
                );
                let raw: serde_json::Value = serde_yaml::from_str(&yaml_requests).unwrap();
                let expanded = crate::expand::expand_config(raw).unwrap();
                let no_flags =
                    crate::flags::ProfileFlags::parse_from_args(&["--concurrency".to_string(), "1".to_string()])
                        .expect("flags parse");
                let run = resolve_expanded_value(
                    expanded,
                    Some("/tmp/x".into()),
                    Some(&no_flags),
                )
                .expect("resolve with YAML requests");
                // Phase should have requests=300 from YAML
                assert_eq!(
                    run.cfg.phases.as_ref().unwrap()[0].common.requests,
                    Some(300),
                    "phase.requests from YAML should be preserved"
                );
            })
            .expect("spawn worker")
            .join()
            .expect("worker panicked");
    }

    /// Every authorable `endpoint:` key must survive YAML -> `Inputs` ->
    /// `Endpoint` -> the protocol-v2 `cfg.endpoint` object.
    ///
    /// `deny_unknown_fields` catches a *misspelled* key, but not a *declared*
    /// one that goes nowhere: each field needs three independent touches — the YAML struct, the `Inputs` projection, and
    /// `resolve::resolve`'s `Endpoint` construction. A field wired into only
    /// the first two reaches the runtime as its default while the config
    /// validates clean; `ssl_verify` and `uds_path` both shipped that way. This
    /// authors every key at a non-default value and asserts the projected
    /// value, so a missing touch fails here rather than at runtime.
    #[test]
    fn every_authored_endpoint_field_reaches_protocol_v2() {
        let cfg = "schemaVersion: \"2.0\"\n\
             benchmark:\n\
            \x20 model: m\n\
            \x20 endpoint:\n\
            \x20   type: chat\n\
            \x20   url: 127.0.0.1:8000\n\
            \x20   streaming: true\n\
            \x20   api_key: sk-authored\n\
            \x20   timeout: 12.5\n\
            \x20   connection_reuse: sticky-user-sessions\n\
            \x20   ssl_verify: false\n\
            \x20   uds_path: /tmp/authored.sock\n\
            \x20   use_legacy_max_tokens: true\n\
            \x20   use_server_token_count: true\n\
            \x20   download_video_content: true\n\
            \x20   headers: {X-Authored: yes}\n\
            \x20   extra: {temperature: 0.25}\n\
            \x20   request_content_type: multipart/form-data\n\
            \x20   session_header: X-Session-Authored\n\
            \x20   proxy: http://proxy.invalid:3128\n\
            \x20   path: /authored/chat\n\
            \x20   wait_for_model_timeout: 7.5\n\
            \x20   wait_for_model_interval: 2.5\n\
            \x20   wait_for_model_mode: both\n\
            \x20   reset_kv_cache: {timeout_seconds: 3.5, path: /reset}\n\
            \x20   server_profiler: {timeout_seconds: 4.5, start_path: /start, stop_path: /stop}\n\
            \x20 dataset: {prompts: {isl: 8, osl: 4}}\n\
            \x20 phases: {type: concurrency, requests: 1, concurrency: 1}\n";
        let run =
            resolve_str(cfg, Some("/tmp/x".into())).expect("fully-authored endpoint resolves");
        let ep = &serde_json::to_value(&run).unwrap()["cfg"]["endpoint"];

        for (key, want) in [
            ("urls", serde_json::json!(["http://127.0.0.1:8000"])),
            ("type", serde_json::json!("chat")),
            ("streaming", serde_json::json!(true)),
            ("api_key", serde_json::json!("sk-authored")),
            ("timeout_seconds", serde_json::json!(12.5)),
            (
                "connection_reuse",
                serde_json::json!("sticky-user-sessions"),
            ),
            ("ssl_verify", serde_json::json!(false)),
            ("uds_path", serde_json::json!("/tmp/authored.sock")),
            ("use_legacy_max_tokens", serde_json::json!(true)),
            ("use_server_token_count", serde_json::json!(true)),
            ("download_video_content", serde_json::json!(true)),
            ("headers", serde_json::json!({"X-Authored": "yes"})),
            ("extra", serde_json::json!({"temperature": 0.25})),
            (
                "request_content_type",
                serde_json::json!("multipart_form_data"),
            ),
            ("session_header", serde_json::json!("X-Session-Authored")),
            ("proxy", serde_json::json!("http://proxy.invalid:3128")),
            ("path", serde_json::json!("/authored/chat")),
            ("wait_for_model_timeout", serde_json::json!(7.5)),
            ("wait_for_model_interval", serde_json::json!(2.5)),
            ("wait_for_model_mode", serde_json::json!("both")),
            (
                "reset_kv_cache",
                serde_json::json!({"timeout_seconds": 3.5, "path": "/reset"}),
            ),
            (
                "server_profiler",
                serde_json::json!({"timeout_seconds": 4.5, "start_path": "/start", "stop_path": "/stop"}),
            ),
        ] {
            assert_eq!(
                ep.get(key),
                Some(&want),
                "endpoint.{key} did not survive YAML -> protocol-v2 projection; \
                 full projected endpoint: {ep:#}"
            );
        }
    }

    /// Every shipped `config/templates/*.yaml` must resolve, exactly as
    /// `aiperf config validate` resolves it.
    ///
    /// The templates are the authoring surface users copy, so a key the loader
    /// drops ships as a benchmark that silently runs something else: six
    /// templates authored `isl`/`osl` directly under `dataset:` and ran the
    /// default sequence lengths, and fourteen authored `excludeFromResults` on
    /// a phase that was derived from the role instead. This runs in-process
    /// against the same `resolve` the CLI calls, so it cannot pass against a
    /// stale binary.
    #[test]
    fn every_shipped_template_resolves() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiperf/config/templates");
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("templates dir {} unreadable: {e}", dir.display()))
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
            .collect();
        entries.sort();
        assert!(
            !entries.is_empty(),
            "no templates found in {}",
            dir.display()
        );

        let failures: Vec<String> = entries
            .iter()
            .filter_map(|p| {
                super::resolve(p, Some("/tmp/aiperf-template-test".into()))
                    .err()
                    .map(|e| format!("  {}: {e:#}", p.file_name().unwrap().to_string_lossy()))
            })
            .collect();
        assert!(
            failures.is_empty(),
            "{} of {} shipped templates do not resolve:\n{}",
            failures.len(),
            entries.len(),
            failures.join("\n")
        );
    }

    /// `dataset.prompts.sequenceDistribution` must reach the protocol-v2
    /// dataset object.
    ///
    /// This is the same three-touch trap as the endpoint fields above: the
    /// runtime consumed `SeqDistEntry` and only `--seq-dist` filled it, so a
    /// YAML-authored mixture parsed clean and then ran the scalar `isl`/`osl`
    /// default instead.
    #[test]
    fn authored_sequence_distribution_reaches_protocol_v2() {
        let cfg = "schemaVersion: \"2.0\"\n\
             benchmark:\n\
            \x20 model: m\n\
            \x20 endpoint: {type: chat, url: 127.0.0.1:8000}\n\
            \x20 dataset:\n\
            \x20   prompts:\n\
            \x20     sequenceDistribution:\n\
            \x20       - {isl: 128, osl: 16, probability: 70}\n\
            \x20       - {isl: 4096, osl: 256, probability: 30}\n\
            \x20 phases: {type: concurrency, requests: 1, concurrency: 1}\n";
        let run = resolve_str(cfg, Some("/tmp/x".into())).expect("seq-dist config resolves");
        let v = serde_json::to_value(&run).unwrap();
        let dist = &v["cfg"]["datasets"][0]["prompts"]["sequence_distribution"];

        let entries = dist
            .as_array()
            .unwrap_or_else(|| panic!("sequence_distribution absent from projection: {dist:#}"));
        assert_eq!(entries.len(), 2, "both mixture entries must survive");
        assert_eq!(entries[0]["probability"], serde_json::json!(70.0));
        assert_eq!(entries[1]["probability"], serde_json::json!(30.0));
        // The scalar forms must project as fixed distributions, not be dropped.
        assert!(
            entries[1]["isl"].to_string().contains("4096"),
            "second entry lost its isl: {:#}",
            entries[1]
        );
    }
}
