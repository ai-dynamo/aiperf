// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `aiperf`: real-HTTP benchmarking CLI.
//!
//! Two modes:
//!
//! ```text
//! # online (default): closed-loop concurrency benchmark
//! aiperf [BASE_URL] [MODEL] --concurrency N --requests N --isl N --osl N
//!
//! # graph: Graph-IR E2E streaming throughput (multi-turn DAG conversations)
//! aiperf --mode graph [BASE_URL] [MODEL] \
//!   --turns N --instances N --workers N --concurrency N --osl N \
//!   [--request-concurrency N] [--http2]
//! ```

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use aiperf::accuracy::{
    AccuracyDataset, grade_and_finalize_accuracy_report, load_evaluator_problems,
};
use aiperf::adaptive::{
    AdaptiveControlVariable, AdaptiveRunConfig, AdaptiveStepConfig, parse_sla_filter,
    positive_seconds_to_ns,
};
use aiperf::agentic::{AgenticWorkload, DatasetAgenticTurnBuilder, finalize_agentic_report};
use aiperf::agentic_gateway::{
    AgenticInferenceGateway, HttpAgenticInferenceGateway, resolve_advertised_host,
};
use aiperf::ancillary::{AncillaryTimingConfig, parse_base_urls};
use aiperf::fixed_schedule::FixedScheduleConfig;
use aiperf::multiturn::{ConversationSource, NativeDatasetConversationSource};
use aiperf::report::{
    print_accuracy_table, print_agentic_table, print_report_table, write_accuracy_summary_csv,
    write_native_report_json, write_scheduled_report_json,
};
use aiperf::run::{
    run_fixed_schedule_online_with_ancillary, run_paced_adaptive_with_metrics_and_ancillary,
    run_scheduled_online, run_single_turn_dataset_online,
    run_user_centric_adaptive_online_with_ancillary, run_user_centric_online_with_ancillary,
};
use aiperf::scheduled::{ScheduledRunReport, TurnRecordProcessor, Workload};
use aiperf::user_centric::UserCentricConfig;
use aiperf::workload::SkeletonWorkload;
use aiperf_accuracy::{
    AccuracyEvaluator, AgenticEvaluator, AgenticEvaluatorLoadConfig, EvaluatorLoadConfig,
    PythonEvaluator, WorkerProcessConfig,
};
use aiperf_adaptive::CorrelationContext;
use aiperf_dataset::{
    ComposeConfig, Dataset, DatasetSource, HuggingFaceTokenizer, LoadConfig,
    SyntheticDatasetConfig, SyntheticPromptConfig, TextTokenizer, TiktokenEncoding,
    TiktokenTokenizer,
};
use aiperf_endpoints::EndpointConfig;
use aiperf_extensions::AiperfRegistry;
use aiperf_metrics::{NativeReport, ReportRunInfo, ReportSummary, RunOutcome};
use aiperf_rng::{RngRoot, SamplingDistribution};
use aiperf_timing::{ArrivalPattern, StopConfig};
use anyhow::Context;
use clap::Parser;

// A high-churn benchmark allocator: the graph executor + streaming client
// allocate heavily per request, and glibc malloc/free was the top profiled
// hotspot. mimalloc cuts that churn substantially.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Default number of conversation instances for `--mode graph`.
const DEFAULT_INSTANCES: usize = 400_000;
/// RPS thresholds for the graph-mode summary verdict lines.
const RPS_1M: f64 = 1_000_000.0;
const RPS_500K: f64 = 500_000.0;
const RPS_300K: f64 = 300_000.0;

/// Command-line arguments for `aiperf`.
///
/// A single top-level struct models both modes: `--mode` selects online
/// (default) or graph, and the field set is the union of the two modes' flags.
/// Numeric flags whose default differs between modes are `Option`s so the
/// per-mode default can be applied in code (matching the legacy parser exactly).
#[derive(Parser, Debug)]
#[command(disable_help_flag = true)]
struct Cli {
    /// Benchmark mode: `online` (default, closed-loop concurrency) or `graph`.
    #[arg(long, default_value = "online")]
    mode: String,

    /// Positional `[BASE_URL]`; online and graph accept a comma-separated list.
    base_url: Option<String>,
    /// Positional `[MODEL]` (default `model`).
    model: Option<String>,

    // --- flags shared between modes (defaults differ, hence Option) ---
    /// Offered concurrency (online default 16, graph default 64).
    #[arg(long)]
    concurrency: Option<usize>,
    /// Output sequence length / max tokens (online default 128, graph default 1).
    #[arg(long)]
    osl: Option<usize>,

    // --- online-only flags ---
    /// Number of requests (online default 100).
    #[arg(long)]
    requests: Option<usize>,
    /// Input sequence length (online default 128).
    #[arg(long)]
    isl: Option<usize>,
    /// Request rate (req/s). When set, runs open-loop request-rate mode with the
    /// `--arrival` pattern instead of closed-loop concurrency. Combine with
    /// `--concurrency` to cap in-flight requests under the rate.
    #[arg(long)]
    request_rate: Option<f64>,
    /// Arrival pattern for `--request-rate`: `poisson` (default), `gamma`, `constant`.
    #[arg(long)]
    arrival: Option<String>,
    /// Gamma burstiness (shape) for `--arrival gamma`; default 1.0 (Poisson-like).
    #[arg(long)]
    smoothness: Option<f64>,
    /// RNG seed for arrival spacing (default 0) — deterministic runs.
    #[arg(long)]
    seed: Option<u64>,
    /// Benchmark duration in seconds. When set (without `--requests`), the run is
    /// bounded by time instead of request count; both set = first-hit wins.
    #[arg(long)]
    duration: Option<f64>,
    /// Write the aggregate report as JSON to this path (online mode).
    #[arg(long)]
    json: Option<PathBuf>,
    /// Write aggregate metrics plus per-turn scheduler timing as JSON.
    #[arg(long)]
    timing_json: Option<PathBuf>,
    /// Select per-user pacing at this aggregate request rate (req/s).
    #[arg(long)]
    user_centric_rate: Option<f64>,
    /// Number of simulated users for `--user-centric-rate`.
    #[arg(long)]
    num_users: Option<usize>,
    /// Stop after starting this many sessions (continuations still drain).
    #[arg(long)]
    sessions: Option<usize>,
    /// JSON/JSONL conversation dataset. Required by `--fixed-schedule`; optional
    /// for user-centric mode, which otherwise uses a synthetic K-turn template.
    #[arg(long)]
    input_file: Option<PathBuf>,
    /// Explicit registered dataset format. Omit for structural detection.
    #[arg(long)]
    input_format: Option<String>,
    /// Dataset tokenizer: built-in tiktoken name, tokenizer.json, or HF directory.
    #[arg(long)]
    tokenizer: Option<String>,
    /// Format-specific option as KEY=JSON; repeat for multiple options.
    #[arg(long = "dataset-option")]
    dataset_options: Vec<String>,
    /// Replay `--input-file` using its trace timestamps.
    #[arg(long)]
    fixed_schedule: bool,
    /// Normalize fixed trace timestamps so the first event lands at run start.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    fixed_schedule_auto_offset: bool,
    /// Inclusive fixed-trace start filter and explicit schedule zero, in ms.
    #[arg(long)]
    fixed_schedule_start_offset_ms: Option<f64>,
    /// Inclusive fixed-trace end filter, in ms.
    #[arg(long)]
    fixed_schedule_end_offset_ms: Option<f64>,
    /// Relative delay between synthetic continuation turns, in ms.
    #[arg(long)]
    think_time_ms: Option<u64>,

    // --- ancillary timing-policy flags (online path) ---
    /// Seconds to ramp session concurrency linearly from one to its target.
    #[arg(long)]
    concurrency_ramp_duration: Option<f64>,
    /// Seconds to ramp prefill concurrency linearly from one to its target.
    #[arg(long)]
    prefill_concurrency_ramp_duration: Option<f64>,
    /// Seconds to ramp request rate from its proportional minimum to target.
    #[arg(long)]
    request_rate_ramp_duration: Option<f64>,
    /// Percentage (0-100) of profiling requests disconnected after send.
    #[arg(long)]
    request_cancellation_rate: Option<f64>,
    /// Seconds after full request send before a selected request is disconnected.
    #[arg(long)]
    request_cancellation_delay: Option<f64>,

    // --- adaptive-scale flags (online path) ---
    /// Enable one-run SLA-driven ramp-until-fail load control.
    #[arg(long)]
    adaptive_scale: bool,
    /// Controller strategy type (only ramp_until_fail is implemented).
    #[arg(long)]
    adaptive_scale_strategy_type: Option<String>,
    /// Control variable: concurrency, prefill_concurrency, request_rate, or users.
    #[arg(long)]
    adaptive_control_variable: Option<String>,
    /// Inclusive adaptive control minimum (default 1).
    #[arg(long)]
    adaptive_control_min: Option<f64>,
    /// Inclusive adaptive control maximum (otherwise inferred from the selected mode).
    #[arg(long)]
    adaptive_control_max: Option<f64>,
    /// Assessment-window duration in seconds (default 30, minimum 1).
    #[arg(long)]
    adaptive_assessment_period: Option<f64>,
    /// Required sustain-hold duration in seconds.
    #[arg(long)]
    adaptive_sustain_duration: Option<f64>,
    /// Minimum successful completions required for a conclusive window.
    #[arg(long)]
    adaptive_min_completed_requests: Option<usize>,
    /// SLA filter in metric:stat:op:threshold form; repeat for conjunctive filters.
    #[arg(long = "adaptive-scale-sla", alias = "adaptive-sla")]
    adaptive_scale_sla: Vec<String>,
    /// Step policy: sla_margin (default) or fixed_percent_step.
    #[arg(long)]
    adaptive_step_policy: Option<String>,
    /// Base increment for sla_margin (default 10).
    #[arg(long)]
    adaptive_base_step: Option<usize>,
    /// Maximum sla_margin base-step multiplier (default 4).
    #[arg(long)]
    adaptive_max_step_multiplier: Option<usize>,
    /// Current-control percentage for fixed_percent_step (default 25).
    #[arg(long)]
    adaptive_step_percent: Option<f64>,
    /// Directory for adaptive_scale_events.jsonl and adaptive_scale_summary.json.
    #[arg(long)]
    adaptive_artifact_dir: Option<PathBuf>,

    // --- accuracy flags (online HTTP path) ---
    /// Run a canonical Python/Lighteval accuracy benchmark through normal Rust inference.
    #[arg(long)]
    accuracy_benchmark: Option<String>,
    /// Benchmark tasks/categories; comma-separated or repeated.
    #[arg(long, value_delimiter = ',')]
    accuracy_tasks: Vec<String>,
    /// Number of few-shot examples; omitted uses the benchmark default.
    #[arg(long)]
    accuracy_n_shots: Option<usize>,
    /// Disable the benchmark's chain-of-thought prompt.
    #[arg(long)]
    accuracy_no_cot: bool,
    /// Enable chain-of-thought even when the benchmark default is off.
    #[arg(long)]
    accuracy_enable_cot: bool,
    /// Override the benchmark's default system prompt.
    #[arg(long)]
    accuracy_system_prompt: Option<String>,
    /// Log every per-problem grading decision after the run.
    #[arg(long)]
    accuracy_verbose: bool,
    /// Deterministically evaluate only the first N selected problems.
    #[arg(long)]
    accuracy_max_problems: Option<usize>,
    /// Override maximum generated tokens per problem.
    #[arg(long)]
    accuracy_max_tokens: Option<usize>,
    /// Input-accounting tokenizer: `builtin`, a tiktoken encoding name, a
    /// tokenizer.json file, or a local Hugging Face model directory.
    #[arg(long)]
    accuracy_tokenizer: Option<String>,
    /// Write the per-task and overall accuracy summary as CSV.
    #[arg(long)]
    accuracy_csv: Option<PathBuf>,

    // --- stateful agentic accuracy flags (online HTTP path) ---
    /// Run a Harbor Hub package, legacy `name@version`, or local task directory.
    #[arg(long)]
    agentic_benchmark: Option<String>,
    /// Exact Harbor task names/globs; comma-separated or repeated.
    #[arg(long, value_delimiter = ',')]
    agentic_tasks: Vec<String>,
    /// Deterministically evaluate only the first N selected task episodes.
    #[arg(long)]
    agentic_max_episodes: Option<usize>,
    /// Maximum Harbor task environments active at once.
    #[arg(long)]
    agentic_task_concurrency: Option<usize>,
    /// Harbor environment provider, for example docker or daytona.
    #[arg(long)]
    agentic_environment: Option<String>,
    /// Directory for canonical Harbor trials and trajectories.
    #[arg(long)]
    agentic_output_dir: Option<PathBuf>,
    /// Maximum Terminus-2 model calls per episode.
    #[arg(long)]
    agentic_max_turns: Option<usize>,
    /// Maximum generated tokens for each Rust-owned inference call.
    #[arg(long)]
    agentic_max_tokens: Option<usize>,
    /// Model context-window size exposed to the canonical agent scaffold.
    #[arg(long)]
    agentic_context_window: Option<usize>,
    /// Canonical Terminus command parser: json or xml.
    #[arg(long)]
    agentic_parser: Option<String>,
    /// Disable canonical Terminus context summarization.
    #[arg(long)]
    agentic_no_summarize: bool,
    /// Verifier reward to report as the primary score.
    #[arg(long)]
    agentic_primary_reward: Option<String>,
    /// Allow Harbor to replace a cached task package.
    #[arg(long)]
    agentic_overwrite: bool,
    /// Input-accounting tokenizer for evaluator-authored model calls.
    #[arg(long)]
    agentic_tokenizer: Option<String>,
    /// Hostname or IP that evaluator sandboxes use to reach Rust's authenticated
    /// auxiliary-inference ingress (default: kernel-selected non-loopback IP).
    #[arg(long)]
    agentic_inference_gateway_host: Option<String>,
    /// Log every canonical episode result after the run.
    #[arg(long)]
    agentic_verbose: bool,

    // --- graph-only flags ---
    /// Conversation turns per instance (graph default 4; online synthetic default 1).
    #[arg(long)]
    turns: Option<usize>,
    /// Conversation instances (graph default 400000).
    #[arg(long)]
    instances: Option<usize>,
    /// Worker threads (graph default: available cores).
    #[arg(long)]
    workers: Option<usize>,
    /// Connections per worker (graph default 8).
    #[arg(long)]
    conns: Option<usize>,
    /// Optional per-request concurrency override (graph).
    #[arg(long)]
    request_concurrency: Option<usize>,
    /// Optional prefill concurrency cap (online) or override (graph).
    #[arg(long)]
    prefill_concurrency: Option<usize>,
    /// Force HTTP/1.1 (graph; accepted for compatibility).
    #[arg(long)]
    http1: bool,
    /// Opt into h2c prior-knowledge (graph).
    #[arg(long)]
    http2: bool,
}

fn main() -> anyhow::Result<()> {
    aiperf::logging::init();
    let cli = Cli::parse();
    let registry = AiperfRegistry::builtin()?;
    ensure_adaptive_flag_envelope(&cli)?;
    ensure_accuracy_flag_envelope(&cli)?;

    match cli.mode.as_str() {
        "graph" => run_graph_mode(&cli),
        "online" if cli.accuracy_benchmark.is_some() => run_accuracy_mode(&cli, &registry),
        "online" if cli.agentic_benchmark.is_some() => run_agentic_mode(&cli, &registry),
        "online" => run_online_mode(&cli, &registry),
        other => anyhow::bail!("unknown --mode '{other}' (expected online|graph)"),
    }
}

fn ensure_accuracy_flag_envelope(cli: &Cli) -> anyhow::Result<()> {
    anyhow::ensure!(
        !(cli.accuracy_enable_cot && cli.accuracy_no_cot),
        "--accuracy-enable-cot conflicts with --accuracy-no-cot"
    );
    anyhow::ensure!(
        !(cli.accuracy_benchmark.is_some() && cli.agentic_benchmark.is_some()),
        "--accuracy-benchmark conflicts with --agentic-benchmark"
    );
    if cli.accuracy_benchmark.is_none() {
        anyhow::ensure!(
            cli.accuracy_tasks.is_empty()
                && cli.accuracy_n_shots.is_none()
                && !cli.accuracy_no_cot
                && !cli.accuracy_enable_cot
                && cli.accuracy_system_prompt.is_none()
                && !cli.accuracy_verbose
                && cli.accuracy_max_problems.is_none()
                && cli.accuracy_max_tokens.is_none()
                && cli.accuracy_tokenizer.is_none()
                && cli.accuracy_csv.is_none(),
            "accuracy options require --accuracy-benchmark"
        );
    }
    if cli.accuracy_benchmark.is_some() {
        anyhow::ensure!(
            !has_agentic_flags(cli),
            "agentic options require --agentic-benchmark and conflict with --accuracy-benchmark"
        );
    }
    if cli.agentic_benchmark.is_none() {
        anyhow::ensure!(
            !has_agentic_flags(cli),
            "agentic options require --agentic-benchmark"
        );
    } else {
        anyhow::ensure!(
            cli.accuracy_tasks.is_empty()
                && cli.accuracy_n_shots.is_none()
                && !cli.accuracy_no_cot
                && !cli.accuracy_enable_cot
                && cli.accuracy_system_prompt.is_none()
                && !cli.accuracy_verbose
                && cli.accuracy_max_problems.is_none()
                && cli.accuracy_max_tokens.is_none()
                && cli.accuracy_tokenizer.is_none()
                && cli.accuracy_csv.is_none(),
            "static accuracy options require --accuracy-benchmark and conflict with --agentic-benchmark"
        );
    }
    Ok(())
}

fn has_agentic_flags(cli: &Cli) -> bool {
    !cli.agentic_tasks.is_empty()
        || cli.agentic_max_episodes.is_some()
        || cli.agentic_task_concurrency.is_some()
        || cli.agentic_environment.is_some()
        || cli.agentic_output_dir.is_some()
        || cli.agentic_max_turns.is_some()
        || cli.agentic_max_tokens.is_some()
        || cli.agentic_context_window.is_some()
        || cli.agentic_parser.is_some()
        || cli.agentic_no_summarize
        || cli.agentic_primary_reward.is_some()
        || cli.agentic_overwrite
        || cli.agentic_tokenizer.is_some()
        || cli.agentic_inference_gateway_host.is_some()
        || cli.agentic_verbose
}

fn ensure_adaptive_flag_envelope(cli: &Cli) -> anyhow::Result<()> {
    if !cli.adaptive_scale {
        anyhow::ensure!(
            cli.adaptive_control_variable.is_none()
                && cli.adaptive_scale_strategy_type.is_none()
                && cli.adaptive_control_min.is_none()
                && cli.adaptive_control_max.is_none()
                && cli.adaptive_assessment_period.is_none()
                && cli.adaptive_sustain_duration.is_none()
                && cli.adaptive_min_completed_requests.is_none()
                && cli.adaptive_scale_sla.is_empty()
                && cli.adaptive_step_policy.is_none()
                && cli.adaptive_base_step.is_none()
                && cli.adaptive_max_step_multiplier.is_none()
                && cli.adaptive_step_percent.is_none()
                && cli.adaptive_artifact_dir.is_none(),
            "adaptive control/SLA flags require --adaptive-scale"
        );
    }
    Ok(())
}

fn run_accuracy_mode(cli: &Cli, registries: &AiperfRegistry) -> anyhow::Result<()> {
    anyhow::ensure!(
        !cli.adaptive_scale,
        "--adaptive-scale is not supported with --accuracy-benchmark"
    );
    anyhow::ensure!(
        !has_ancillary_timing_flags(cli),
        "ancillary timing-policy flags are not supported with --accuracy-benchmark"
    );
    let base_url = cli
        .base_url
        .clone()
        .unwrap_or_else(|| "http://localhost:8000".to_string());
    let model = cli.model.clone().unwrap_or_else(|| "model".to_string());
    let requested_benchmark = cli
        .accuracy_benchmark
        .as_deref()
        .expect("caller checked accuracy benchmark");
    let enable_cot = if cli.accuracy_no_cot {
        Some(false)
    } else if cli.accuracy_enable_cot {
        Some(true)
    } else {
        None
    };
    if let Some(n_shots) = cli.accuracy_n_shots {
        anyhow::ensure!(n_shots <= 32, "--accuracy-n-shots must be at most 32");
    }
    if let Some(limit) = cli.accuracy_max_problems {
        anyhow::ensure!(
            limit > 0,
            "--accuracy-max-problems must be greater than zero"
        );
    }
    if let Some(max_tokens) = cli.accuracy_max_tokens {
        anyhow::ensure!(
            max_tokens > 0,
            "--accuracy-max-tokens must be greater than zero"
        );
    }
    let evaluator_config = EvaluatorLoadConfig {
        tasks: (!cli.accuracy_tasks.is_empty()).then(|| cli.accuracy_tasks.clone()),
        n_shots: cli.accuracy_n_shots,
        enable_cot,
        system_prompt: cli.accuracy_system_prompt.clone(),
        max_problems: cli.accuracy_max_problems,
        max_tokens: cli.accuracy_max_tokens,
        seed: cli.seed.unwrap_or(0),
    };
    let concurrency = cli.concurrency.unwrap_or(16);
    anyhow::ensure!(concurrency > 0, "--concurrency must be greater than zero");

    tracing::info!(
        benchmark = requested_benchmark,
        base = %base_url,
        model = %model,
        concurrency,
        tasks = ?cli.accuracy_tasks,
        n_shots = ?cli.accuracy_n_shots,
        enable_cot = ?enable_cot,
        max_problems = ?cli.accuracy_max_problems,
        "starting canonical accuracy benchmark"
    );

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    let report = local.block_on(&rt, async {
        let mut evaluator = PythonEvaluator::spawn(WorkerProcessConfig::python_module())
            .await
            .context("starting canonical Python accuracy evaluator")?;
        tracing::info!(
            protocol = evaluator.identity().protocol,
            worker_version = evaluator.identity().worker_version,
            python_version = evaluator.identity().python_version,
            python_executable = evaluator.identity().python_executable,
            packages = ?evaluator.identity().packages,
            dependency_lock_sha256 = ?evaluator.identity().dependency_lock_sha256,
            container_digest = ?evaluator.identity().container_digest,
            "canonical accuracy evaluator initialized"
        );
        let result = async {
            let (loaded, problems) =
                load_evaluator_problems(&mut evaluator, requested_benchmark, &evaluator_config)
                    .await?;
            let tokenizer = load_tokenizer(cli.accuracy_tokenizer.as_deref())?;
            let dataset =
                AccuracyDataset::from_evaluator_problems(&model, problems, tokenizer.as_ref())?;
            tracing::info!(
                benchmark = loaded.benchmark,
                grader = loaded.grader,
                dataset = ?loaded.dataset,
                problems = dataset.len(),
                tokenizer = tokenizer.name(),
                segments = dataset.dataset().segments().len(),
                "canonical accuracy problems materialized for normal inference"
            );
            let processor = std::rc::Rc::new(dataset.record_processor());
            let source: Box<dyn ConversationSource> = Box::new(
                NativeDatasetConversationSource::sequential_with_endpoint_config_and_resolver(
                    dataset.dataset().as_ref().clone(),
                    model.clone(),
                    cli.accuracy_max_tokens.unwrap_or(2_048),
                    EndpointConfig {
                        streaming: true,
                        use_legacy_max_tokens: true,
                        use_server_token_count: true,
                        ..EndpointConfig::default()
                    },
                    registries.endpoint_resolver(),
                )?,
            );
            let processors: Vec<std::rc::Rc<dyn TurnRecordProcessor>> = vec![processor.clone()];
            let scheduled = run_single_turn_dataset_online(
                base_url.clone(),
                model.clone(),
                source,
                concurrency,
                cli.http2,
                processors,
            )
            .await?;
            grade_and_finalize_accuracy_report(
                &model,
                scheduled,
                &dataset,
                processor.as_ref(),
                &mut evaluator,
                &loaded,
            )
            .await
        }
        .await;
        match result {
            Ok(report) => {
                evaluator
                    .shutdown()
                    .await
                    .context("shutting down canonical accuracy evaluator")?;
                Ok(report)
            }
            Err(error) => {
                if let Err(shutdown_error) = evaluator.shutdown().await {
                    tracing::warn!(
                        error = %shutdown_error,
                        "accuracy evaluator also failed during error-path shutdown"
                    );
                }
                Err(error)
            }
        }
    })?;
    print_report_table(&report.performance);
    print_accuracy_table(&report.accuracy);
    if cli.accuracy_verbose {
        for record in &report.records {
            tracing::info!(
                correlation_id = record.correlation_id.as_str(),
                task = record.task.as_str(),
                correct = record.result.correct,
                unparsed = record.result.unparsed,
                extracted = ?record.result.extracted,
                reasoning = ?record.result.reasoning,
                "accuracy grading result"
            );
        }
    }
    if let Some(path) = &cli.json {
        write_native_report_json(&report.native_report, path)?;
    }
    if let Some(path) = &cli.accuracy_csv {
        write_accuracy_summary_csv(&report.accuracy, path)?;
    }
    Ok(())
}

fn run_agentic_mode(cli: &Cli, registries: &AiperfRegistry) -> anyhow::Result<()> {
    anyhow::ensure!(
        !cli.adaptive_scale,
        "--adaptive-scale is not supported with --agentic-benchmark"
    );
    anyhow::ensure!(
        !has_ancillary_timing_flags(cli),
        "ancillary timing-policy flags are not supported with --agentic-benchmark"
    );
    anyhow::ensure!(
        cli.request_rate.is_none()
            && cli.user_centric_rate.is_none()
            && !cli.fixed_schedule
            && cli.input_file.is_none()
            && cli.requests.is_none()
            && cli.duration.is_none()
            && cli.sessions.is_none()
            && cli.prefill_concurrency.is_none(),
        "--agentic-benchmark owns its task schedule and conflicts with ordinary workload/stop flags"
    );
    let base_url = cli
        .base_url
        .clone()
        .unwrap_or_else(|| "http://localhost:8000".to_string());
    let model = cli.model.clone().unwrap_or_else(|| "model".to_string());
    let requested_dataset = cli
        .agentic_benchmark
        .as_deref()
        .expect("caller checked agentic benchmark");
    let model_concurrency = cli.concurrency.unwrap_or(16);
    let task_concurrency = cli.agentic_task_concurrency.unwrap_or(1);
    let max_tokens = cli.agentic_max_tokens.unwrap_or(4_096);
    let context_window = cli.agentic_context_window.unwrap_or(131_072);
    let parser = cli.agentic_parser.as_deref().unwrap_or("json");
    let inference_gateway_host =
        resolve_advertised_host(cli.agentic_inference_gateway_host.as_deref())?;
    anyhow::ensure!(
        model_concurrency > 0,
        "--concurrency must be greater than zero"
    );
    anyhow::ensure!(
        task_concurrency > 0,
        "--agentic-task-concurrency must be greater than zero"
    );
    anyhow::ensure!(
        max_tokens > 0,
        "--agentic-max-tokens must be greater than zero"
    );
    anyhow::ensure!(
        context_window > 0,
        "--agentic-context-window must be greater than zero"
    );
    anyhow::ensure!(
        max_tokens <= context_window,
        "--agentic-max-tokens must not exceed --agentic-context-window"
    );
    anyhow::ensure!(
        matches!(parser, "json" | "xml"),
        "--agentic-parser must be json or xml"
    );
    for (name, value) in [
        ("--agentic-max-episodes", cli.agentic_max_episodes),
        ("--agentic-max-turns", cli.agentic_max_turns),
    ] {
        if let Some(value) = value {
            anyhow::ensure!(value > 0, "{name} must be greater than zero");
        }
    }
    let output_dir = cli
        .agentic_output_dir
        .as_deref()
        .unwrap_or_else(|| std::path::Path::new("artifacts/agentic"))
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("--agentic-output-dir must be valid UTF-8"))?
        .to_string();
    let evaluator_config = AgenticEvaluatorLoadConfig {
        task_names: (!cli.agentic_tasks.is_empty()).then(|| cli.agentic_tasks.clone()),
        max_episodes: cli.agentic_max_episodes,
        task_concurrency,
        environment: cli
            .agentic_environment
            .clone()
            .unwrap_or_else(|| "docker".to_string()),
        output_dir,
        max_turns: cli.agentic_max_turns,
        max_tokens,
        context_window,
        parser: parser.to_string(),
        enable_summarize: !cli.agentic_no_summarize,
        primary_reward: cli.agentic_primary_reward.clone(),
        overwrite: cli.agentic_overwrite,
        inference_gateway: None,
    };

    tracing::info!(
        dataset = requested_dataset,
        base = %base_url,
        model = %model,
        model_concurrency,
        task_concurrency,
        environment = evaluator_config.environment,
        max_episodes = ?evaluator_config.max_episodes,
        max_turns = ?evaluator_config.max_turns,
        "starting canonical agentic accuracy benchmark"
    );

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    let report = local.block_on(&rt, async {
        let evaluator = PythonEvaluator::spawn(WorkerProcessConfig::python_module())
            .await
            .context("starting canonical Python agentic evaluator")?;
        anyhow::ensure!(
            evaluator.supports_agentic(),
            "canonical evaluator worker does not advertise agentic_harbor support"
        );
        let worker_identity = evaluator.identity().clone();
        tracing::info!(
            protocol = worker_identity.protocol,
            worker_version = worker_identity.worker_version,
            python_version = worker_identity.python_version,
            python_executable = worker_identity.python_executable,
            packages = ?worker_identity.packages,
            dependency_lock_sha256 = ?worker_identity.dependency_lock_sha256,
            container_digest = ?worker_identity.container_digest,
            "canonical agentic evaluator initialized"
        );
        let tokenizer: Arc<dyn TextTokenizer> =
            Arc::from(load_tokenizer(cli.agentic_tokenizer.as_deref())?);
        let turn_builder = Rc::new(DatasetAgenticTurnBuilder::new(
            model.clone(),
            tokenizer.clone(),
            EndpointConfig {
                streaming: true,
                use_legacy_max_tokens: true,
                use_server_token_count: true,
                ..EndpointConfig::default()
            },
            registries.endpoint_resolver(),
        )?);
        let evaluator: Box<dyn AgenticEvaluator> = Box::new(evaluator);
        let inference_gateway =
            HttpAgenticInferenceGateway::bind(&inference_gateway_host, max_tokens)
                .await
                .context("starting Rust agentic auxiliary-inference ingress")?;
        tracing::info!(
            base_url = inference_gateway.evaluator_config().base_url,
            "Rust agentic auxiliary-inference ingress initialized"
        );
        let workload = AgenticWorkload::prepare_with_gateway(
            evaluator,
            requested_dataset,
            &model,
            &evaluator_config,
            model_concurrency,
            turn_builder,
            Some(Box::new(inference_gateway)),
        )
        .await?;
        let evaluator_identity = workload.identity().clone();
        tracing::info!(
            harness = evaluator_identity.harness,
            harness_version = evaluator_identity.harness_version,
            harness_source_sha256 = evaluator_identity.harness_source_sha256,
            dataset = ?evaluator_identity.dataset,
            agent = evaluator_identity.agent,
            agent_version = evaluator_identity.agent_version,
            environment = evaluator_identity.environment,
            verifier = evaluator_identity.verifier,
            episodes = evaluator_identity.episode_count,
            primary_reward = ?evaluator_identity.primary_reward,
            tokenizer = tokenizer.name(),
            "canonical agentic tasks frozen before measurement"
        );
        let result = async {
            let scheduled_workload: Rc<dyn Workload> = workload.clone();
            let scheduled = run_scheduled_online(
                base_url.clone(),
                model.clone(),
                scheduled_workload,
                cli.http2,
                Vec::new(),
            )
            .await?;
            let results = workload.results()?;
            finalize_agentic_report(
                requested_dataset,
                &model,
                model_concurrency,
                scheduled,
                worker_identity,
                evaluator_identity,
                workload.config(),
                results,
            )
        }
        .await;
        match result {
            Ok(report) => {
                workload.shutdown().await?;
                Ok(report)
            }
            Err(error) => {
                if let Err(shutdown_error) = workload.shutdown().await {
                    tracing::warn!(
                        error = %shutdown_error,
                        "agentic evaluator also failed during error-path shutdown"
                    );
                }
                Err(error)
            }
        }
    })?;

    print_report_table(&report.performance);
    print_agentic_table(&report.evaluation.summary);
    if report.evaluation.summary.infrastructure_error_count > 0
        || report.evaluation.summary.cancelled_count > 0
    {
        tracing::warn!(
            infrastructure_errors = report.evaluation.summary.infrastructure_error_count,
            cancelled = report.evaluation.summary.cancelled_count,
            "non-scored agentic episodes are reported as infrastructure/cancellation, not incorrect answers"
        );
    }
    if cli.agentic_verbose {
        for result in &report.results {
            tracing::info!(
                episode_id = result.episode_id.as_str(),
                task = result.task,
                outcome = ?result.outcome,
                rewards = ?result.rewards,
                primary_reward = ?result.primary_reward,
                model_calls = result.model_calls,
                primary_model_calls = result.primary_model_calls,
                environment_model_calls = result.environment_model_calls,
                verifier_model_calls = result.verifier_model_calls,
                error_kind = ?result.error_kind,
                error_message = ?result.error_message,
                artifact_path = ?result.artifact_path,
                "agentic canonical episode result"
            );
        }
    }
    if let Some(path) = &cli.json {
        write_native_report_json(&report.native_report, path)?;
    }
    Ok(())
}

fn load_tokenizer(spec: Option<&str>) -> anyhow::Result<Box<dyn TextTokenizer>> {
    let spec = spec.unwrap_or("builtin");
    let path = std::path::Path::new(spec);
    if path.is_dir() {
        return Ok(Box::new(HuggingFaceTokenizer::from_directory(path)?));
    }
    if path.is_file() {
        return Ok(Box::new(HuggingFaceTokenizer::from_file(path)?));
    }
    let encoding = spec.parse::<TiktokenEncoding>()?;
    Ok(Box::new(TiktokenTokenizer::new(encoding)))
}

fn parse_dataset_options(cli: &Cli) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let mut options = serde_json::Map::new();
    for option in &cli.dataset_options {
        let (key, authored) = option
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--dataset-option must be KEY=JSON, got {option:?}"))?;
        anyhow::ensure!(!key.trim().is_empty(), "dataset option key cannot be empty");
        let value = serde_json::from_str(authored)
            .unwrap_or_else(|_| serde_json::Value::String(authored.to_string()));
        options.insert(key.trim().to_string(), value);
    }
    Ok(options)
}

struct DatasetBuildContext<'a> {
    registries: &'a AiperfRegistry,
    runtime: &'a tokio::runtime::Runtime,
    local: &'a tokio::task::LocalSet,
    model: &'a str,
}

fn load_native_file_dataset(
    cli: &Cli,
    context: &DatasetBuildContext<'_>,
    osl: usize,
) -> anyhow::Result<Dataset> {
    let path = cli
        .input_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("a dataset file was not supplied"))?;
    let tokenizer = load_tokenizer(cli.tokenizer.as_deref())?;
    let options = parse_dataset_options(cli)?;
    let mut load = LoadConfig::new(DatasetSource::Path(path.clone()));
    load.options = options.clone();
    if cli.fixed_schedule {
        load.options
            .insert("fixed_schedule".into(), serde_json::Value::Bool(true));
    }
    let mut compose = ComposeConfig::new(context.model, RngRoot::new(cli.seed));
    compose.output_length_distribution = Some(SamplingDistribution::fixed(osl as f64)?);
    compose.format_options = options;
    compose
        .format_options
        .entry("max_conversations")
        .or_insert_with(|| {
            serde_json::Value::from(cli.sessions.or(cli.requests).unwrap_or(100).max(1) as u64)
        });
    context
        .local
        .block_on(
            context.runtime,
            context.registries.dataset_formats().build_dataset(
                cli.input_format.as_deref(),
                &load,
                &compose,
                tokenizer.as_ref(),
            ),
        )
        .map_err(Into::into)
}

fn build_native_synthetic_dataset(
    cli: &Cli,
    context: &DatasetBuildContext<'_>,
    isl: usize,
    osl: usize,
    turns: usize,
) -> anyhow::Result<Dataset> {
    let tokenizer = load_tokenizer(cli.tokenizer.as_deref())?;
    let mut compose = ComposeConfig::new(context.model, RngRoot::new(cli.seed));
    compose.output_length_distribution = Some(SamplingDistribution::fixed(osl as f64)?);
    compose.synthetic_config = Some(SyntheticDatasetConfig {
        entries: cli.sessions.or(cli.requests).unwrap_or(100).max(1),
        turns: SamplingDistribution::fixed(turns.max(1) as f64)?,
        turn_delay_ms: SamplingDistribution::fixed(cli.think_time_ms.unwrap_or(0) as f64)?,
        prompts: Some(SyntheticPromptConfig {
            input_tokens: SamplingDistribution::fixed(isl as f64)?,
            batch_size: 1,
        }),
        ..SyntheticDatasetConfig::default()
    });
    let load = LoadConfig::new(DatasetSource::Inline(
        serde_json::json!({"__aiperf_synthetic": true}),
    ));
    context
        .local
        .block_on(
            context.runtime,
            context.registries.dataset_formats().build_dataset(
                Some("synthetic"),
                &load,
                &compose,
                tokenizer.as_ref(),
            ),
        )
        .map_err(Into::into)
}

fn run_online_mode(cli: &Cli, registries: &AiperfRegistry) -> anyhow::Result<()> {
    let base_url = cli
        .base_url
        .clone()
        .unwrap_or_else(|| "http://localhost:8000".to_string());
    let model = cli.model.clone().unwrap_or_else(|| "model".to_string());

    let concurrency = cli.concurrency.unwrap_or(16usize);
    let num_requests = cli.requests.unwrap_or(100usize);
    let isl = cli.isl.unwrap_or(128usize);
    let osl = cli.osl.unwrap_or(128usize);
    let ancillary = build_ancillary_timing_config(cli)?;

    anyhow::ensure!(isl > 0, "--isl must be greater than zero");
    anyhow::ensure!(osl > 0, "--osl must be greater than zero");
    anyhow::ensure!(concurrency > 0, "--concurrency must be greater than zero");
    if let Some(requests) = cli.requests {
        anyhow::ensure!(requests > 0, "--requests must be greater than zero");
    }
    if let Some(sessions) = cli.sessions {
        anyhow::ensure!(sessions > 0, "--sessions must be greater than zero");
    }
    if let Some(duration) = cli.duration {
        anyhow::ensure!(
            duration.is_finite() && duration > 0.0,
            "--duration must be positive and finite"
        );
    }
    if let Some(rate) = cli.request_rate {
        anyhow::ensure!(
            rate.is_finite() && rate > 0.0,
            "--request-rate must be positive and finite"
        );
    }
    if let Some(prefill) = cli.prefill_concurrency {
        anyhow::ensure!(
            prefill > 0,
            "--prefill-concurrency must be greater than zero"
        );
    }
    anyhow::ensure!(
        cli.input_format.is_none() || cli.input_file.is_some(),
        "--input-format requires --input-file"
    );
    anyhow::ensure!(
        cli.dataset_options.is_empty() || cli.input_file.is_some(),
        "--dataset-option requires --input-file"
    );
    anyhow::ensure!(
        cli.tokenizer.is_none() || cli.input_file.is_some() || cli.user_centric_rate.is_some(),
        "--tokenizer requires --input-file or --user-centric-rate"
    );
    if cli.fixed_schedule {
        anyhow::ensure!(
            cli.user_centric_rate.is_none() && cli.request_rate.is_none(),
            "--fixed-schedule conflicts with --user-centric-rate and --request-rate"
        );
        anyhow::ensure!(
            cli.concurrency.is_none(),
            "--fixed-schedule is pure open-loop replay and does not accept --concurrency"
        );
        anyhow::ensure!(
            !cli.adaptive_scale,
            "--fixed-schedule conflicts with --adaptive-scale"
        );
        anyhow::ensure!(
            !ancillary.has_ramps(),
            "fixed-schedule replay has authored timestamps and does not accept actuator ramps"
        );
    } else {
        anyhow::ensure!(
            cli.fixed_schedule_start_offset_ms.is_none()
                && cli.fixed_schedule_end_offset_ms.is_none(),
            "fixed-schedule offsets require --fixed-schedule"
        );
    }
    if cli.user_centric_rate.is_some() {
        anyhow::ensure!(
            cli.request_rate.is_none(),
            "--user-centric-rate conflicts with --request-rate"
        );
    }

    tracing::info!(
        base = %base_url,
        model = %model,
        concurrency,
        requests = num_requests,
        isl,
        osl,
        "starting aiperf online benchmark"
    );

    // The online sink is `!Send` (hyper transport over `Rc<dyn Clock>`), so drive
    // the run on a current-thread runtime + LocalSet.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let local = tokio::task::LocalSet::new();
    let dataset_build = DatasetBuildContext {
        registries,
        runtime: &rt,
        local: &local,
        model: &model,
    };

    if cli.fixed_schedule {
        let input_file = cli
            .input_file
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--fixed-schedule requires --input-file"))?;
        anyhow::ensure!(
            !(cli.fixed_schedule_auto_offset && cli.fixed_schedule_start_offset_ms.is_some()),
            "--fixed-schedule-auto-offset=true conflicts with --fixed-schedule-start-offset-ms"
        );
        let _ = input_file;
        let dataset = load_native_file_dataset(cli, &dataset_build, osl)?
            .filter_first_turn_window(
                cli.fixed_schedule_start_offset_ms,
                cli.fixed_schedule_end_offset_ms,
            )?;
        let source: Box<dyn ConversationSource> =
            Box::new(NativeDatasetConversationSource::preferred_with_registries(
                dataset,
                model.clone(),
                osl,
                RngRoot::new(cli.seed),
                registries.samplers(),
                registries.endpoint_resolver(),
            )?);
        let report = local.block_on(
            &rt,
            run_fixed_schedule_online_with_ancillary(
                base_url,
                model,
                source,
                FixedScheduleConfig {
                    auto_offset_timestamps: cli.fixed_schedule_auto_offset,
                    start_offset_ms: cli.fixed_schedule_start_offset_ms,
                },
                cli.http2,
                ancillary,
                cli.seed.unwrap_or(0),
            ),
        )?;
        emit_scheduled_report(cli, &report)?;
        return Ok(());
    }

    // Stop bounds: explicit `--requests` wins the count cap; else if `--duration` is
    // or `--sessions` is set, omit the default count; else default to 100.
    let duration_ns = cli
        .duration
        .map(|seconds| (seconds * 1_000_000_000.0).round_ties_even() as i64);
    let count: Option<u64> = if let Some(r) = cli.requests {
        Some(r as u64)
    } else if duration_ns.is_some() || cli.sessions.is_some() {
        None
    } else {
        Some(100)
    };
    let stop = StopConfig {
        total_expected_requests: count,
        expected_num_sessions: cli.sessions.map(|sessions| sessions as u64),
        expected_duration_ns: duration_ns,
    };

    if let Some(rate) = cli.user_centric_rate {
        let num_users = cli
            .num_users
            .ok_or_else(|| anyhow::anyhow!("--user-centric-rate requires --num-users"))?;
        anyhow::ensure!(num_users > 0, "--num-users must be greater than zero");
        anyhow::ensure!(
            rate.is_finite() && rate > 0.0,
            "--user-centric-rate must be positive and finite"
        );
        let adaptive =
            build_adaptive_cli_config(cli, concurrency, ArrivalPattern::ConcurrencyBurst)?;
        if let Some(config) = &adaptive {
            anyhow::ensure!(
                config.control_variable == AdaptiveControlVariable::Users,
                "--user-centric-rate adaptive scale requires --adaptive-control-variable users"
            );
            anyhow::ensure!(
                config.minimum.fract() == 0.0,
                "adaptive users minimum must be an integer"
            );
        }
        let initial_users = adaptive
            .as_ref()
            .map_or(num_users, |config| config.minimum as usize);
        if let Some(requests) = cli.requests {
            anyhow::ensure!(
                requests >= num_users,
                "--requests ({requests}) must be >= --num-users ({num_users})"
            );
        }
        if let Some(sessions) = cli.sessions {
            anyhow::ensure!(
                sessions >= num_users,
                "--sessions ({sessions}) must be >= --num-users ({num_users})"
            );
        }
        let turns = cli.turns.unwrap_or(4);
        anyhow::ensure!(
            turns >= 2 || cli.input_file.is_some(),
            "user-centric synthetic mode requires --turns >= 2"
        );
        let source: Box<dyn ConversationSource> = if cli.input_file.is_some() {
            let dataset = load_native_file_dataset(cli, &dataset_build, osl)?;
            Box::new(NativeDatasetConversationSource::preferred_with_registries(
                dataset,
                model.clone(),
                osl,
                RngRoot::new(cli.seed),
                registries.samplers(),
                registries.endpoint_resolver(),
            )?)
        } else {
            let dataset = build_native_synthetic_dataset(cli, &dataset_build, isl, osl, turns)?;
            Box::new(NativeDatasetConversationSource::preferred_with_registries(
                dataset,
                model.clone(),
                osl,
                RngRoot::new(cli.seed),
                registries.samplers(),
                registries.endpoint_resolver(),
            )?)
        };
        let user_config = UserCentricConfig {
            num_users: initial_users,
            request_rate: rate,
            concurrency: cli.concurrency,
        };
        let report = match adaptive {
            Some(adaptive) => local.block_on(
                &rt,
                run_user_centric_adaptive_online_with_ancillary(
                    base_url,
                    model,
                    source,
                    user_config,
                    stop,
                    cli.http2,
                    adaptive,
                    ancillary,
                    cli.seed.unwrap_or(0),
                ),
            )?,
            None => local.block_on(
                &rt,
                run_user_centric_online_with_ancillary(
                    base_url,
                    model,
                    source,
                    user_config,
                    stop,
                    cli.http2,
                    ancillary,
                    cli.seed.unwrap_or(0),
                ),
            )?,
        };
        emit_scheduled_report(cli, &report)?;
        return Ok(());
    }

    anyhow::ensure!(
        cli.timing_json.is_none(),
        "--timing-json is only valid with --fixed-schedule or --user-centric-rate"
    );
    anyhow::ensure!(
        cli.num_users.is_none(),
        "--num-users requires --user-centric-rate"
    );
    anyhow::ensure!(
        cli.input_file.is_none(),
        "--input-file is currently supported by --fixed-schedule and --user-centric-rate"
    );

    let workload = SkeletonWorkload {
        num_requests,
        input_tokens: isl,
        output_tokens: osl,
        turns: 1,
        think_time_ms: None,
    };

    // `--request-rate` selects open-loop rate mode; absent = closed-loop concurrency
    // (ConcurrencyBurst) which defaults to concurrency 16.
    let (pattern, rate) = match cli.request_rate {
        Some(r) => {
            let p = match cli.arrival.as_deref() {
                Some("constant") => ArrivalPattern::Constant,
                Some("gamma") => ArrivalPattern::Gamma,
                Some("poisson") | None => ArrivalPattern::Poisson,
                Some(other) => {
                    anyhow::bail!("unknown --arrival '{other}' (expected constant|poisson|gamma)")
                }
            };
            (p, Some(r))
        }
        None => (ArrivalPattern::ConcurrencyBurst, None),
    };
    let concurrency_opt = if cli.request_rate.is_some() {
        cli.concurrency // open-loop unless capped
    } else {
        Some(concurrency) // closed-loop concurrency
    };

    let adaptive = build_adaptive_cli_config(cli, concurrency, pattern)?;
    anyhow::ensure!(
        !matches!(
            adaptive.as_ref().map(|config| config.control_variable),
            Some(AdaptiveControlVariable::Users)
        ),
        "adaptive users requires --user-centric-rate"
    );

    let report_endpoints = parse_base_urls(&base_url)?;
    let report_model = model.clone();
    let report = local.block_on(
        &rt,
        run_paced_adaptive_with_metrics_and_ancillary(
            base_url,
            model,
            workload,
            pattern,
            rate,
            cli.smoothness,
            concurrency_opt,
            cli.prefill_concurrency,
            stop,
            cli.seed.unwrap_or(0),
            adaptive,
            ancillary,
        ),
    )?;
    print_report_table(&report.performance);
    if let Some(path) = &cli.json {
        let successful = if report.performance.request_counts.completed_requests > 0 {
            report_endpoints.clone()
        } else {
            Vec::new()
        };
        let native_report = NativeReport::from_outcome(
            &report.metrics,
            &RunOutcome {
                run: ReportRunInfo {
                    mode: Some("online".to_string()),
                    model: Some(report_model),
                },
                summary: ReportSummary {
                    endpoints_configured: report_endpoints,
                    endpoints_successful: successful,
                    ..ReportSummary::default()
                },
                ..RunOutcome::default()
            },
        );
        write_native_report_json(&native_report, path)?;
    }
    Ok(())
}

fn has_ancillary_timing_flags(cli: &Cli) -> bool {
    cli.concurrency_ramp_duration.is_some()
        || cli.prefill_concurrency_ramp_duration.is_some()
        || cli.request_rate_ramp_duration.is_some()
        || cli.request_cancellation_rate.is_some()
        || cli.request_cancellation_delay.is_some()
}

fn build_ancillary_timing_config(cli: &Cli) -> anyhow::Result<AncillaryTimingConfig> {
    let positive_duration = |value: Option<f64>, flag: &str| -> anyhow::Result<Option<u64>> {
        value
            .map(|seconds| {
                let ns = positive_seconds_to_ns(seconds, flag)?;
                Ok(u64::try_from(ns).expect("positive nanoseconds fit u64"))
            })
            .transpose()
    };
    if let Some(rate) = cli.request_cancellation_rate {
        anyhow::ensure!(
            rate.is_finite() && (0.0..=100.0).contains(&rate),
            "--request-cancellation-rate must be a finite percentage in 0..=100"
        );
    }
    anyhow::ensure!(
        cli.request_cancellation_delay.is_none() || cli.request_cancellation_rate.is_some(),
        "--request-cancellation-delay requires --request-cancellation-rate"
    );
    let cancellation_delay_ns = match cli.request_cancellation_delay {
        Some(seconds) => {
            anyhow::ensure!(
                seconds.is_finite() && seconds >= 0.0,
                "--request-cancellation-delay must be finite and non-negative"
            );
            let ns = seconds * 1_000_000_000.0;
            anyhow::ensure!(
                ns <= i64::MAX as f64,
                "--request-cancellation-delay is outside the i64 nanosecond range"
            );
            // Python uses `int(delay * NANOS_PER_SECOND)`, which truncates.
            ns as i64
        }
        None => 0,
    };
    let config = AncillaryTimingConfig {
        concurrency_ramp_duration_ns: positive_duration(
            cli.concurrency_ramp_duration,
            "--concurrency-ramp-duration",
        )?,
        prefill_concurrency_ramp_duration_ns: positive_duration(
            cli.prefill_concurrency_ramp_duration,
            "--prefill-concurrency-ramp-duration",
        )?,
        request_rate_ramp_duration_ns: positive_duration(
            cli.request_rate_ramp_duration,
            "--request-rate-ramp-duration",
        )?,
        cancellation_rate_percent: cli.request_cancellation_rate,
        cancellation_delay_ns,
        ..AncillaryTimingConfig::default()
    };
    config.validate()?;
    Ok(config)
}

fn build_adaptive_cli_config(
    cli: &Cli,
    concurrency: usize,
    pattern: ArrivalPattern,
) -> anyhow::Result<Option<AdaptiveRunConfig>> {
    if !cli.adaptive_scale {
        return Ok(None);
    }

    anyhow::ensure!(
        cli.duration.is_some(),
        "--adaptive-scale requires --duration"
    );
    let sustain_duration = cli
        .adaptive_sustain_duration
        .ok_or_else(|| anyhow::anyhow!("--adaptive-scale requires --adaptive-sustain-duration"))?;
    anyhow::ensure!(
        !cli.adaptive_scale_sla.is_empty(),
        "--adaptive-scale requires at least one --adaptive-scale-sla"
    );
    let strategy_type = cli
        .adaptive_scale_strategy_type
        .as_deref()
        .unwrap_or("ramp_until_fail");
    anyhow::ensure!(
        strategy_type == "ramp_until_fail",
        "unknown --adaptive-scale-strategy-type {strategy_type:?} (expected ramp_until_fail)"
    );
    let control_variable = cli
        .adaptive_control_variable
        .as_deref()
        .unwrap_or("concurrency")
        .parse::<AdaptiveControlVariable>()?;
    let minimum = cli.adaptive_control_min.unwrap_or(1.0);
    let maximum = if let Some(maximum) = cli.adaptive_control_max {
        maximum
    } else {
        match control_variable {
            AdaptiveControlVariable::Concurrency => concurrency as f64,
            AdaptiveControlVariable::PrefillConcurrency => cli
                .prefill_concurrency
                .map(|value| value as f64)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "adaptive prefill_concurrency requires --prefill-concurrency or --adaptive-control-max"
                    )
                })?,
            AdaptiveControlVariable::RequestRate => cli.request_rate.ok_or_else(|| {
                anyhow::anyhow!("adaptive request_rate requires --request-rate")
            })?,
            AdaptiveControlVariable::Users => cli
                .num_users
                .map(|value| value as f64)
                .ok_or_else(|| anyhow::anyhow!("adaptive users requires --num-users"))?,
        }
    };
    if control_variable == AdaptiveControlVariable::RequestRate {
        anyhow::ensure!(
            pattern != ArrivalPattern::ConcurrencyBurst && cli.request_rate.is_some(),
            "adaptive request_rate requires --request-rate"
        );
    }
    anyhow::ensure!(
        minimum.is_finite() && minimum > 0.0,
        "--adaptive-control-min must be positive and finite"
    );
    anyhow::ensure!(
        maximum.is_finite() && maximum > minimum,
        "--adaptive-control-max ({maximum}) must be finite and > minimum ({minimum})"
    );
    if matches!(
        control_variable,
        AdaptiveControlVariable::Concurrency
            | AdaptiveControlVariable::PrefillConcurrency
            | AdaptiveControlVariable::Users
    ) {
        anyhow::ensure!(
            minimum.fract() == 0.0
                && maximum.fract() == 0.0
                && minimum <= usize::MAX as f64
                && maximum <= usize::MAX as f64,
            "adaptive {control_variable:?} control bounds must be integers in the usize range"
        );
    }
    if control_variable == AdaptiveControlVariable::PrefillConcurrency {
        let session_limit = if pattern == ArrivalPattern::ConcurrencyBurst {
            Some(concurrency)
        } else {
            cli.concurrency
        }
        .ok_or_else(|| {
            anyhow::anyhow!("adaptive prefill_concurrency requires a session --concurrency cap")
        })?;
        anyhow::ensure!(
            maximum <= session_limit as f64,
            "adaptive prefill_concurrency maximum must be <= concurrency"
        );
    }
    anyhow::ensure!(
        cli.adaptive_min_completed_requests.unwrap_or(1) > 0,
        "--adaptive-min-completed-requests must be >= 1"
    );

    let step = match cli.adaptive_step_policy.as_deref().unwrap_or("sla_margin") {
        "sla_margin" => {
            let base_step = cli.adaptive_base_step.unwrap_or(10);
            let max_step_multiplier = cli.adaptive_max_step_multiplier.unwrap_or(4);
            anyhow::ensure!(base_step > 0, "--adaptive-base-step must be >= 1");
            anyhow::ensure!(
                max_step_multiplier > 0,
                "--adaptive-max-step-multiplier must be >= 1"
            );
            AdaptiveStepConfig::SlaMargin {
                base_step,
                max_step_multiplier,
            }
        }
        "fixed_percent_step" | "fixed-percent-step" => {
            let percent = cli.adaptive_step_percent.unwrap_or(25.0);
            anyhow::ensure!(
                percent.is_finite() && percent > 0.0,
                "--adaptive-step-percent must be positive and finite"
            );
            AdaptiveStepConfig::FixedPercent { percent }
        }
        other => anyhow::bail!(
            "unknown --adaptive-step-policy {other:?} (expected sla_margin|fixed_percent_step)"
        ),
    };
    let artifact_dir = cli.adaptive_artifact_dir.clone().unwrap_or_else(|| {
        cli.json
            .as_ref()
            .and_then(|path| path.parent())
            .filter(|path| !path.as_os_str().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
    });
    let assessment_period = cli.adaptive_assessment_period.unwrap_or(30.0);
    anyhow::ensure!(
        assessment_period.is_finite() && assessment_period >= 1.0,
        "--adaptive-assessment-period must be finite and >= 1 second"
    );
    Ok(Some(AdaptiveRunConfig {
        control_variable,
        minimum,
        maximum,
        assessment_period_ns: positive_seconds_to_ns(
            assessment_period,
            "--adaptive-assessment-period",
        )?,
        sustain_duration_ns: positive_seconds_to_ns(
            sustain_duration,
            "--adaptive-sustain-duration",
        )?,
        min_completed_requests: cli.adaptive_min_completed_requests.unwrap_or(1),
        sla_filters: cli
            .adaptive_scale_sla
            .iter()
            .map(|value| parse_sla_filter(value))
            .collect::<anyhow::Result<Vec<_>>>()?,
        step,
        artifact_dir,
        correlation: CorrelationContext {
            phase_id: "profiling".to_string(),
            phase_name: Some("profiling".to_string()),
            ..Default::default()
        },
    }))
}

fn emit_scheduled_report(cli: &Cli, report: &ScheduledRunReport) -> anyhow::Result<()> {
    print_report_table(&report.performance);
    println!(
        "Schedule timing : {} turns, mean lateness {:.3} ms, max lateness {:.3} ms",
        report.schedule_timing.issued_turns,
        report.schedule_timing.mean_issue_lateness_ms,
        report.schedule_timing.max_issue_lateness_ms,
    );
    if let Some(path) = &cli.json {
        let endpoint = cli
            .base_url
            .clone()
            .unwrap_or_else(|| "http://localhost:8000".to_string());
        let endpoints = parse_base_urls(&endpoint)?;
        let successful = if report.performance.request_counts.completed_requests > 0 {
            endpoints.clone()
        } else {
            Vec::new()
        };
        let native_report = NativeReport::from_outcome(
            &report.native_metrics,
            &RunOutcome {
                run: ReportRunInfo {
                    mode: Some(format!("online:{}", report.strategy)),
                    model: Some(cli.model.clone().unwrap_or_else(|| "model".to_string())),
                },
                summary: ReportSummary {
                    endpoints_configured: endpoints,
                    endpoints_successful: successful,
                    ..ReportSummary::default()
                },
                ..RunOutcome::default()
            },
        );
        write_native_report_json(&native_report, path)?;
    }
    if let Some(path) = &cli.timing_json {
        write_scheduled_report_json(report, path)?;
    }
    Ok(())
}

/// Parsed graph-mode invocation: the bench config plus transport-selection
/// knobs and the raw `base_url` string (retained for the startup banner).
struct GraphParams {
    cfg: aiperf_graph::bench::BenchConfig,
    base_url: String,
    http2: bool,
    conns: usize,
}

/// Aggregated graph-mode results, formatted by [`print_graph_summary`].
struct GraphSummary {
    rps: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    mean: f64,
    secs: f64,
    extra: String,
}

/// Parse a [`Cli`] into a [`GraphParams`] (positionals + flags → bench config).
fn parse_graph_config(cli: &Cli) -> GraphParams {
    use aiperf_graph::bench::BenchConfig;

    let base_url = cli
        .base_url
        .clone()
        .unwrap_or_else(|| "http://127.0.0.1:8000".to_string());
    let model = cli.model.clone().unwrap_or_else(|| "model".to_string());

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let turns: usize = cli.turns.unwrap_or(4);
    let workers: usize = cli.workers.unwrap_or(cores);
    let concurrency: usize = cli.concurrency.unwrap_or(64);
    let max_tokens: usize = cli.osl.unwrap_or(1);
    // A `--duration` bound governs the run; when the user did not also pin
    // `--instances`, let the duration decide by leaving the instance pool
    // effectively unbounded. Otherwise fall back to the fixed default count.
    let max_duration_ns: Option<i64> = cli.duration.map(|s| (s * 1_000_000_000.0) as i64);
    let instances: usize = cli.instances.unwrap_or(if max_duration_ns.is_some() {
        usize::MAX
    } else {
        DEFAULT_INSTANCES
    });
    let request_concurrency: Option<usize> = cli.request_concurrency;
    let prefill_concurrency: Option<usize> = cli.prefill_concurrency;
    // Transport default is HTTP/1.1 keep-alive: for serial per-lane requests it
    // outperforms h2c (no per-stream hpack/flow-control overhead). --http2 opts
    // into h2c prior-knowledge (multiplexed pool).
    let http2 = cli.http2;
    // `--http1` is accepted for compatibility (it was a silent no-op in the
    // legacy parser too): HTTP/1.1 keep-alive is already the transport default,
    // so the flag has no additional effect beyond not passing `--http2`.
    let _http1 = cli.http1;
    let conns: usize = cli.conns.unwrap_or(8);

    let cfg = BenchConfig {
        base_urls: base_url.split(',').map(|s| s.trim().to_string()).collect(),
        model,
        turns,
        instances,
        workers,
        concurrency,
        max_tokens,
        request_concurrency,
        prefill_concurrency,
        max_duration_ns,
    };

    GraphParams {
        cfg,
        base_url,
        http2,
        conns,
    }
}

/// Render the graph-mode summary to stdout (byte-exact with the legacy output).
fn print_graph_summary(s: &GraphSummary, backend: &str) {
    let rps = s.rps;
    println!("\n=== aiperf --mode graph (Graph-IR E2E, streaming SSE, backend={backend}) ===");
    println!("{}", s.extra);
    println!("wall        : {:.3} s", s.secs);
    println!("RPS         : {rps:.0} req/s");
    println!("TTFT p50    : {:.3} ms", s.p50);
    println!("TTFT p90    : {:.3} ms", s.p90);
    println!("TTFT p99    : {:.3} ms", s.p99);
    println!("TTFT mean   : {:.3} ms", s.mean);
    if rps >= RPS_1M {
        println!("\nPROVEN: aiperf --mode graph >= 1M req/s ({rps:.0}, backend={backend})");
    } else if rps >= RPS_500K {
        println!("\nPROVEN: aiperf --mode graph >= 500k req/s ({rps:.0}, backend={backend})");
    } else if rps >= RPS_300K {
        println!("\n>= 300k: {rps:.0}");
    } else {
        println!("\nbelow 300k: {rps:.0}");
    }
}

fn run_graph_mode(cli: &Cli) -> anyhow::Result<()> {
    anyhow::ensure!(
        !cli.adaptive_scale,
        "--adaptive-scale is supported by online workloads, not --mode graph"
    );
    anyhow::ensure!(
        !has_ancillary_timing_flags(cli),
        "ancillary timing-policy flags are supported by online workloads, not --mode graph"
    );
    use aiperf_graph::transport_bench::run_transport_bench;

    let GraphParams {
        cfg,
        base_url,
        http2,
        conns,
    } = parse_graph_config(cli);

    let backend = "aiperf-transport";
    let report_model = cfg.model.clone();
    let report_endpoints = cfg.base_urls.clone();

    tracing::info!(
        backend,
        base = %base_url,
        turns = cfg.turns,
        instances = cfg.instances,
        workers = cfg.workers,
        concurrency = cfg.concurrency,
        osl = cfg.max_tokens,
        conns_per_worker = conns,
        offered_concurrency = cfg.workers * cfg.concurrency,
        http2,
        "starting aiperf graph benchmark"
    );

    let r = run_transport_bench(cfg, http2, conns);
    let summary = GraphSummary {
        rps: r.rps(),
        p50: r.ttft_p50_ms,
        p90: r.ttft_p90_ms,
        p99: r.ttft_p99_ms,
        mean: r.ttft_mean_ms,
        secs: r.wall_secs,
        extra: format!(
            "completed={} errors={} output_tokens={} output_tok/s={:.0}",
            r.completed,
            r.errors,
            r.output_tokens,
            r.output_tps()
        ),
    };

    print_graph_summary(&summary, backend);
    if let Some(path) = &cli.json {
        let successful = if r.completed > 0 {
            report_endpoints.clone()
        } else {
            Vec::new()
        };
        let native_report = NativeReport::from_outcome(
            &r.native_metrics,
            &RunOutcome {
                run: ReportRunInfo {
                    mode: Some("graph".to_string()),
                    model: Some(report_model),
                },
                summary: ReportSummary {
                    duration_s: Some(r.wall_secs),
                    endpoints_configured: report_endpoints,
                    endpoints_successful: successful,
                    ..ReportSummary::default()
                },
                ..RunOutcome::default()
            },
        );
        write_native_report_json(&native_report, path)?;
    }
    Ok(())
}
