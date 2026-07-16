// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The `aiperf profile` CLI flag surface (clap), byte-exact to the Python names.
//!
//! This is the CLI half of the pre-translation input contract. Flag long-names
//! and aliases mirror `src/aiperf/config/flags/cli_config.py` (which in turn
//! keeps GenAI-Perf aliases). The loader ([`crate::load`]) maps these into the
//! native [`crate::model::BenchmarkRun`].
//!
//! Only the flags needed for the single-run synthetic path are modeled today;
//! the surface grows as sections are ported. Sweep-capable numeric flags are
//! taken as `String` so a comma-list (`--concurrency 10,20,30`) can be detected
//! and rejected as multi-run rather than mis-parsed.

use std::path::PathBuf;

use clap::Parser;

/// Parsed `aiperf profile` flags.
#[derive(Debug, Clone, Parser)]
#[command(name = "profile", disable_help_subcommand = true)]
pub struct ProfileFlags {
    /// Model name(s) to benchmark (`--model-names` / `--model` / `-m`).
    #[arg(
        long = "model-names",
        visible_alias = "model",
        short = 'm',
        value_delimiter = ',',
        num_args = 1..,
    )]
    pub model_names: Vec<String>,

    /// Base URL(s) of the server(s) (`--url` / `-u`).
    #[arg(long = "url", short = 'u', num_args = 1..)]
    pub urls: Vec<String>,

    /// Endpoint dialect id (`--endpoint-type`).
    #[arg(long = "endpoint-type")]
    pub endpoint_type: Option<String>,

    /// Tokenizer identity (`--tokenizer`); defaults to the primary model.
    #[arg(long = "tokenizer")]
    pub tokenizer: Option<String>,

    /// Tokenizer revision / git ref (`--tokenizer-revision`).
    #[arg(long = "tokenizer-revision")]
    pub tokenizer_revision: Option<String>,

    /// Trust remote tokenizer code (`--tokenizer-trust-remote-code`).
    #[arg(long = "tokenizer-trust-remote-code", default_value_t = false)]
    pub tokenizer_trust_remote_code: bool,

    /// Number of conversations to generate (`--num-conversations`); comma ⇒ sweep.
    #[arg(long = "num-conversations", visible_alias = "conversation-num")]
    pub num_conversations: Option<String>,

    /// Explicit synthetic dataset entry count (`--num-dataset-entries` /
    /// `--num-prompts`); highest precedence for the dataset size (over
    /// `--num-conversations` and `--request-count`).
    #[arg(long = "num-dataset-entries", visible_alias = "num-prompts")]
    pub num_dataset_entries: Option<String>,

    /// Custom request path appended to the endpoint URL (`--custom-endpoint` /
    /// `--endpoint`), e.g. `/v2/chat`.
    #[arg(long = "custom-endpoint", visible_alias = "endpoint")]
    pub custom_endpoint: Option<String>,

    /// Per-record export level (`--export-level` / `--profile-export-level`):
    /// `summary` (no per-record files), `records` (JSONL, default), or `raw`
    /// (JSONL + raw request/response JSONL).
    #[arg(long = "export-level", visible_alias = "profile-export-level")]
    pub export_level: Option<String>,

    /// Emit per-request HTTP trace columns (`--export-http-trace`).
    #[arg(long = "export-http-trace", default_value_t = false)]
    pub export_http_trace: bool,

    /// Cellular (multi-process) cell count (`--cells`); `1` = single process.
    #[arg(long = "cells")]
    pub cells: Option<u32>,

    /// Mixed ISL/OSL sequence distribution (`--seq-dist` / `--sequence-distribution`),
    /// e.g. `256,128:60;512,256:40` (optional stddev: `256|10,128|5:60`).
    #[arg(long = "seq-dist", visible_alias = "sequence-distribution")]
    pub seq_dist: Option<String>,

    /// Multi-value sweep combination strategy (`--sweep-type`): `grid` (Cartesian
    /// product, default) or `zip` (element-wise lockstep).
    #[arg(long = "sweep-type", default_value = "grid")]
    pub sweep_type: String,

    /// Suppress the live sweep summary table (`--no-sweep-table`).
    #[arg(long = "no-sweep-table", default_value_t = false)]
    pub no_sweep_table: bool,

    /// Named submission scenario (`--scenario`; `cfg.scenario`).
    #[arg(long = "scenario")]
    pub scenario: Option<String>,

    /// Recorded-graph trajectory-start window lower ratio (`--trajectory-start-min-ratio`).
    #[arg(long = "trajectory-start-min-ratio")]
    pub trajectory_start_min_ratio: Option<f64>,

    /// Recorded-graph trajectory-start window upper ratio (`--trajectory-start-max-ratio`).
    #[arg(long = "trajectory-start-max-ratio")]
    pub trajectory_start_max_ratio: Option<f64>,

    /// Relax cross-field config validation (`--unsafe-override`).
    #[arg(long = "unsafe-override", default_value_t = false)]
    pub unsafe_override: bool,

    /// Agentic cache-warmup duration, seconds (`--agentic-cache-warmup-duration`);
    /// auto-creates a minimal warmup phase carrying it.
    #[arg(long = "agentic-cache-warmup-duration")]
    pub agentic_cache_warmup_duration: Option<f64>,

    /// Rankings: passages per request mean (`--rankings-passages-mean`).
    #[arg(long = "rankings-passages-mean")]
    pub rankings_passages_mean: Option<f64>,
    /// Rankings: passages per request stddev (`--rankings-passages-stddev`).
    #[arg(long = "rankings-passages-stddev")]
    pub rankings_passages_stddev: Option<f64>,
    /// Rankings: per-passage token mean (`--rankings-passages-prompt-token-mean`).
    #[arg(long = "rankings-passages-prompt-token-mean")]
    pub rankings_passages_prompt_token_mean: Option<f64>,
    /// Rankings: per-passage token stddev (`--rankings-passages-prompt-token-stddev`).
    #[arg(long = "rankings-passages-prompt-token-stddev")]
    pub rankings_passages_prompt_token_stddev: Option<f64>,
    /// Rankings: query token mean (`--rankings-query-prompt-token-mean`).
    #[arg(long = "rankings-query-prompt-token-mean")]
    pub rankings_query_prompt_token_mean: Option<f64>,
    /// Rankings: query token stddev (`--rankings-query-prompt-token-stddev`).
    #[arg(long = "rankings-query-prompt-token-stddev")]
    pub rankings_query_prompt_token_stddev: Option<f64>,

    /// Accuracy benchmark id (`--accuracy-benchmark`, e.g. `mmlu`); enables the
    /// accuracy block. The runner owns accuracy execution.
    #[arg(long = "accuracy-benchmark")]
    pub accuracy_benchmark: Option<String>,
    /// Accuracy task ids (`--accuracy-tasks`, repeatable).
    #[arg(long = "accuracy-tasks", num_args = 1..)]
    pub accuracy_tasks: Option<Vec<String>>,
    /// Accuracy grader id (`--accuracy-grader`).
    #[arg(long = "accuracy-grader")]
    pub accuracy_grader: Option<String>,
    /// Few-shot example count (`--accuracy-n-shots`).
    #[arg(long = "accuracy-n-shots")]
    pub accuracy_n_shots: Option<i64>,
    /// Accuracy system prompt override (`--accuracy-system-prompt`).
    #[arg(long = "accuracy-system-prompt")]
    pub accuracy_system_prompt: Option<String>,
    /// Enable chain-of-thought (`--accuracy-enable-cot`).
    #[arg(
        long = "accuracy-enable-cot",
        overrides_with = "accuracy_no_enable_cot"
    )]
    pub accuracy_enable_cot: bool,
    /// Disable chain-of-thought (`--accuracy-no-enable-cot`).
    #[arg(long = "accuracy-no-enable-cot")]
    pub accuracy_no_enable_cot: bool,
    /// Verbose grader output (`--accuracy-verbose`).
    #[arg(long = "accuracy-verbose", default_value_t = false)]
    pub accuracy_verbose: bool,

    /// Auto-set a consistent run seed (default; `--no-set-consistent-seed` disables).
    #[arg(
        long = "set-consistent-seed",
        default_value_t = true,
        overrides_with = "no_set_consistent_seed"
    )]
    pub set_consistent_seed: bool,
    /// Disable the auto consistent seed (`--no-set-consistent-seed`).
    #[arg(long = "no-set-consistent-seed")]
    pub no_set_consistent_seed: bool,
    /// Use the same seed for every sweep variation (`--parameter-sweep-same-seed`).
    #[arg(
        long = "parameter-sweep-same-seed",
        default_value_t = false,
        overrides_with = "no_parameter_sweep_same_seed"
    )]
    pub parameter_sweep_same_seed: bool,
    /// Inverse of the above (`--no-parameter-sweep-same-seed`).
    #[arg(long = "no-parameter-sweep-same-seed")]
    pub no_parameter_sweep_same_seed: bool,
    /// Derive a distinct seed per trial (`--vary-seed-per-trial`).
    #[arg(
        long = "vary-seed-per-trial",
        default_value_t = false,
        overrides_with = "no_vary_seed_per_trial"
    )]
    pub vary_seed_per_trial: bool,
    /// Inverse of the above (`--no-vary-seed-per-trial`).
    #[arg(long = "no-vary-seed-per-trial")]
    pub no_vary_seed_per_trial: bool,
    /// Confidence level for the sweep aggregate (`--confidence-level`, default 0.95).
    #[arg(long = "confidence-level")]
    pub confidence_level: Option<f64>,
    /// Cooldown between sweep variations, seconds (`--parameter-sweep-cooldown-seconds`).
    #[arg(long = "parameter-sweep-cooldown-seconds")]
    pub parameter_sweep_cooldown_seconds: Option<f64>,

    /// Trace synthesis: timestamp speedup ratio (`--synthesis-speedup-ratio`).
    #[arg(long = "synthesis-speedup-ratio")]
    pub synthesis_speedup_ratio: Option<f64>,
    /// Trace synthesis: prefix branch length multiplier (`--synthesis-prefix-len-multiplier`).
    #[arg(long = "synthesis-prefix-len-multiplier")]
    pub synthesis_prefix_len_multiplier: Option<f64>,
    /// Trace synthesis: radix-tree root count multiplier (`--synthesis-prefix-root-multiplier`).
    #[arg(long = "synthesis-prefix-root-multiplier")]
    pub synthesis_prefix_root_multiplier: Option<i64>,
    /// Trace synthesis: prompt (leaf) length multiplier (`--synthesis-prompt-len-multiplier`).
    #[arg(long = "synthesis-prompt-len-multiplier")]
    pub synthesis_prompt_len_multiplier: Option<f64>,
    /// Trace synthesis: output length multiplier (`--synthesis-output-len-multiplier`).
    #[arg(long = "synthesis-output-len-multiplier")]
    pub synthesis_output_len_multiplier: Option<f64>,
    /// Trace synthesis: max input length filter (`--synthesis-max-isl`).
    #[arg(long = "synthesis-max-isl")]
    pub synthesis_max_isl: Option<i64>,
    /// Trace synthesis: max output length filter (`--synthesis-max-osl`).
    #[arg(long = "synthesis-max-osl")]
    pub synthesis_max_osl: Option<i64>,
    /// Trace synthesis: idle-gap cap, seconds (`--synthesis-idle-gap-cap`).
    #[arg(long = "synthesis-idle-gap-cap")]
    pub synthesis_idle_gap_cap: Option<f64>,

    /// URL-selection strategy for multi-URL runs (`--url-strategy`, default
    /// `round_robin`). A runtime knob Python does not project into the request;
    /// accepted for CLI parity.
    #[arg(long = "url-strategy")]
    pub url_strategy: Option<String>,
    /// Disable auto fixed-schedule detection (`--no-fixed-schedule`). A trace-only
    /// toggle with no request effect on the common path; accepted for parity.
    #[arg(long = "no-fixed-schedule", default_value_t = false)]
    pub no_fixed_schedule: bool,

    /// Public-dataset loader filters as `key=value` (`--dataset-filter`, repeatable).
    /// Requires `--public-dataset`; merged into the dataset's loader options.
    #[arg(long = "dataset-filter", num_args = 1..)]
    pub dataset_filter: Option<Vec<String>>,

    // --- Adaptive search / sweep-space (Tier 3) ------------------------------
    /// Search recipe id (`--search-recipe`, e.g. `concurrency-ramp`).
    #[arg(long = "search-recipe")]
    pub search_recipe: Option<String>,
    /// Concurrency search lower bound (`--concurrency-min`).
    #[arg(long = "concurrency-min")]
    pub concurrency_min: Option<i64>,
    /// Concurrency search upper bound (`--concurrency-max`).
    #[arg(long = "concurrency-max")]
    pub concurrency_max: Option<i64>,
    /// Concurrency search step count (`--concurrency-steps`).
    #[arg(long = "concurrency-steps")]
    pub concurrency_steps: Option<i64>,
    /// ISL search lower bound (`--isl-min`).
    #[arg(long = "isl-min")]
    pub isl_min: Option<i64>,
    /// ISL search upper bound (`--isl-max`).
    #[arg(long = "isl-max")]
    pub isl_max: Option<i64>,
    /// ISL search step count (`--isl-steps`).
    #[arg(long = "isl-steps")]
    pub isl_steps: Option<i64>,
    /// OSL search lower bound (`--osl-min`).
    #[arg(long = "osl-min")]
    pub osl_min: Option<i64>,
    /// OSL search upper bound (`--osl-max`).
    #[arg(long = "osl-max")]
    pub osl_max: Option<i64>,
    /// OSL search step count (`--osl-steps`).
    #[arg(long = "osl-steps")]
    pub osl_steps: Option<i64>,
    /// Paired ISL:OSL workload shapes (`--isl-osl-pairs`, repeatable `isl:osl`).
    #[arg(long = "isl-osl-pairs", num_args = 1..)]
    pub isl_osl_pairs: Option<Vec<String>>,
    /// Degradation-knee metric tag (`--degradation-metric-tag`).
    #[arg(long = "degradation-metric-tag")]
    pub degradation_metric_tag: Option<String>,
    /// Degradation-knee stat (`--degradation-stat`).
    #[arg(long = "degradation-stat")]
    pub degradation_stat: Option<String>,
    /// Degradation-knee threshold fraction (`--degradation-threshold`).
    #[arg(long = "degradation-threshold")]
    pub degradation_threshold: Option<f64>,
    /// Paired ISL/OSL shapes for pareto-sweep (`--isl-osl-pairs`, `isl/osl` list).
    // (declared above via `isl_osl_pairs`)
    /// Adaptive search style (`--search-style`: smooth_isotonic/monotonic/bo/optuna/grid).
    #[arg(long = "search-style")]
    pub search_style: Option<String>,
    /// Search planner id (`--search-planner`).
    #[arg(long = "search-planner")]
    pub search_planner: Option<String>,
    /// Search objective metric (`--search-metric`).
    #[arg(long = "search-metric")]
    pub search_metric: Option<String>,
    /// Search objective stat (`--search-stat`).
    #[arg(long = "search-stat")]
    pub search_stat: Option<String>,
    /// Search optimization direction (`--search-direction`).
    #[arg(long = "search-direction")]
    pub search_direction: Option<String>,
    /// Search space definition (`--search-space`).
    #[arg(long = "search-space")]
    pub search_space: Option<String>,
    /// Search SLA filters (`--search-sla`).
    #[arg(long = "search-sla", num_args = 1..)]
    pub search_sla: Option<Vec<String>>,
    /// Search SLA tier (`--search-sla-tier`).
    #[arg(long = "search-sla-tier")]
    pub search_sla_tier: Option<String>,
    /// Initial BO sample count (`--search-initial-points`).
    #[arg(long = "search-initial-points")]
    pub search_initial_points: Option<i64>,
    /// Max search iterations (`--search-max-iterations`).
    #[arg(long = "search-max-iterations")]
    pub search_max_iterations: Option<i64>,
    /// Search random seed (`--search-random-seed`).
    #[arg(long = "search-random-seed")]
    pub search_random_seed: Option<u64>,
    /// Percentile pooling mode for search (`--search-percentile-pooling`).
    #[arg(long = "search-percentile-pooling")]
    pub search_percentile_pooling: Option<String>,
    /// Convergence metric (`--convergence-metric`).
    #[arg(long = "convergence-metric")]
    pub convergence_metric: Option<String>,
    /// Convergence mode (`--convergence-mode`).
    #[arg(long = "convergence-mode")]
    pub convergence_mode: Option<String>,
    /// Convergence stat (`--convergence-stat`).
    #[arg(long = "convergence-stat")]
    pub convergence_stat: Option<String>,
    /// Convergence threshold (`--convergence-threshold`).
    #[arg(long = "convergence-threshold")]
    pub convergence_threshold: Option<f64>,
    /// Optuna sampler (`--optuna-sampler`).
    #[arg(long = "optuna-sampler")]
    pub optuna_sampler: Option<String>,
    /// Optuna acquisition function (`--optuna-acquisition`).
    #[arg(long = "optuna-acquisition")]
    pub optuna_acquisition: Option<String>,
    /// Optuna terminator (`--optuna-terminator`).
    #[arg(long = "optuna-terminator")]
    pub optuna_terminator: Option<String>,
    /// BO constraint mode (`--bo-constraint-mode`).
    #[arg(long = "bo-constraint-mode")]
    pub bo_constraint_mode: Option<String>,
    /// Adaptive-scale control variable (`--adaptive-scale-control`).
    #[arg(long = "adaptive-scale-control")]
    pub adaptive_scale_control: Option<String>,
    /// Named scenario variant selector (`--variant` / `--sweep-variant`).
    #[arg(long = "variant", visible_alias = "sweep-variant")]
    pub variant: Option<String>,
    /// TTFT SLA, ms (`--ttft-sla-ms`).
    #[arg(long = "ttft-sla-ms")]
    pub ttft_sla_ms: Option<f64>,
    /// ITL SLA, ms (`--itl-sla-ms`).
    #[arg(long = "itl-sla-ms")]
    pub itl_sla_ms: Option<f64>,
    /// TPOT SLA, ms (`--tpot-sla-ms`).
    #[arg(long = "tpot-sla-ms")]
    pub tpot_sla_ms: Option<f64>,
    /// End-to-end SLA, ms (`--e2e-sla-ms`).
    #[arg(long = "e2e-sla-ms")]
    pub e2e_sla_ms: Option<f64>,
    /// Error-rate SLA fraction (`--error-rate-sla`).
    #[arg(long = "error-rate-sla")]
    pub error_rate_sla: Option<f64>,
    /// SLO attainment fraction (`--slo-attainment-fraction`).
    #[arg(long = "slo-attainment-fraction")]
    pub slo_attainment_fraction: Option<f64>,
    /// Enable OTel live-streaming (`--stream`; requires `--otel-url`).
    #[arg(long = "stream", default_value_t = false)]
    pub stream: bool,

    // --- Service / UI / logging flags (accepted for CLI parity; the native
    //     entry point does not project these into the runner request) ----------
    /// Controller API host (`--api-host`).
    #[arg(long = "api-host")]
    pub api_host: Option<String>,
    /// Controller API port (`--api-port`).
    #[arg(long = "api-port")]
    pub api_port: Option<u16>,
    /// Log level (`--log-level`).
    #[arg(long = "log-level")]
    pub log_level: Option<String>,
    /// Verbose logging (`--verbose`).
    #[arg(long = "verbose", short = 'v', default_value_t = false)]
    pub verbose: bool,
    /// Extra-verbose logging (`--extra-verbose`).
    #[arg(long = "extra-verbose", default_value_t = false)]
    pub extra_verbose: bool,
    /// UI type (`--ui-type` / `--ui`).
    #[arg(long = "ui-type", visible_alias = "ui")]
    pub ui_type: Option<String>,
    /// Max worker processes (`--workers-max` / `--max-workers`).
    #[arg(long = "workers-max", visible_alias = "max-workers")]
    pub workers_max: Option<u32>,
    /// Record-processor service count (`--record-processor-service-count` / `--record-processors`).
    #[arg(
        long = "record-processor-service-count",
        visible_alias = "record-processors"
    )]
    pub record_processor_service_count: Option<u32>,
    /// Live stats refresh interval, seconds (`--stats-interval`).
    #[arg(long = "stats-interval")]
    pub stats_interval: Option<f64>,
    /// Show per-request trace timing (`--show-trace-timing`).
    #[arg(long = "show-trace-timing", default_value_t = false)]
    pub show_trace_timing: bool,
    /// Auto-generate plots after the run (`--auto-plot` / `--no-auto-plot`).
    #[arg(
        long = "auto-plot",
        overrides_with = "no_auto_plot",
        default_value_t = false
    )]
    pub auto_plot: bool,
    /// Inverse of `--auto-plot`.
    #[arg(long = "no-auto-plot", default_value_t = false)]
    pub no_auto_plot: bool,
    /// Fail the run if plotting fails (`--plot-required`).
    #[arg(long = "plot-required", default_value_t = false)]
    pub plot_required: bool,
    /// Export file prefix (`--profile-export-prefix` / `--profile-export-file`).
    #[arg(long = "profile-export-prefix", visible_alias = "profile-export-file")]
    pub profile_export_prefix: Option<String>,
    /// ZMQ TCP host (`--zmq-host`).
    #[arg(long = "zmq-host")]
    pub zmq_host: Option<String>,
    /// ZMQ IPC path (`--zmq-ipc-path`).
    #[arg(long = "zmq-ipc-path")]
    pub zmq_ipc_path: Option<String>,
    /// ZMQ dual-bind mode (`--zmq-dual-bind`).
    #[arg(long = "zmq-dual-bind", default_value_t = false)]
    pub zmq_dual_bind: bool,

    /// Trials per variation (`--num-profile-runs`); `>1` repeats each run.
    #[arg(long = "num-profile-runs")]
    pub num_profile_runs: Option<u32>,

    /// Trial iteration order (`--parameter-sweep-mode`): `repeated` (trials outer,
    /// default) or `independent` (variations outer).
    #[arg(long = "parameter-sweep-mode", default_value = "repeated")]
    pub parameter_sweep_mode: String,

    /// Seconds to wait between trials/variations (`--profile-run-cooldown-seconds`).
    #[arg(long = "profile-run-cooldown-seconds")]
    pub profile_run_cooldown_seconds: Option<f64>,

    /// Keep the warmup phase on trials after the first
    /// (`--no-profile-run-disable-warmup-after-first`); default drops it.
    #[arg(
        long = "profile-run-disable-warmup-after-first",
        default_value_t = true,
        overrides_with = "no_profile_run_disable_warmup_after_first"
    )]
    pub profile_run_disable_warmup_after_first: bool,
    /// Inverse of the above (keep warmup on every trial).
    #[arg(long = "no-profile-run-disable-warmup-after-first")]
    pub no_profile_run_disable_warmup_after_first: bool,

    /// Mean turns per session for multi-turn (`--session-turns-mean`).
    #[arg(long = "session-turns-mean", visible_alias = "conversation-turn-mean")]
    pub session_turns_mean: Option<f64>,

    /// Stddev of turns per session (`--session-turns-stddev`).
    #[arg(
        long = "session-turns-stddev",
        visible_alias = "conversation-turn-stddev"
    )]
    pub session_turns_stddev: Option<f64>,

    /// Number of sessions to generate (`--num-sessions`); comma ⇒ sweep.
    #[arg(long = "num-sessions")]
    pub num_sessions: Option<String>,

    /// Per-session think-time delay ratio (`--session-delay-ratio`).
    #[arg(
        long = "session-delay-ratio",
        visible_alias = "conversation-turn-delay-ratio"
    )]
    pub session_delay_ratio: Option<f64>,

    /// Mean inter-turn delay, milliseconds (`--session-turn-delay-mean`).
    #[arg(
        long = "session-turn-delay-mean",
        visible_alias = "conversation-turn-delay-mean"
    )]
    pub session_turn_delay_mean: Option<f64>,

    /// Stddev of inter-turn delay, milliseconds (`--session-turn-delay-stddev`).
    #[arg(
        long = "session-turn-delay-stddev",
        visible_alias = "conversation-turn-delay-stddev"
    )]
    pub session_turn_delay_stddev: Option<f64>,

    /// Per-session affinity header name (`--session-header`).
    #[arg(long = "session-header")]
    pub session_header: Option<String>,

    /// Number of warmup sessions (`--num-warmup-sessions`).
    #[arg(long = "num-warmup-sessions")]
    pub num_warmup_sessions: Option<u64>,

    /// Warmup prefill concurrency (`--warmup-prefill-concurrency`).
    #[arg(long = "warmup-prefill-concurrency")]
    pub warmup_prefill_concurrency: Option<u32>,

    /// Warmup arrival distribution (`--warmup-arrival-pattern`;
    /// `poisson`/`gamma`/`constant`).
    #[arg(long = "warmup-arrival-pattern")]
    pub warmup_arrival_pattern: Option<String>,

    /// Warmup prefill-concurrency-ramp duration
    /// (`--warmup-prefill-concurrency-ramp-duration`).
    #[arg(long = "warmup-prefill-concurrency-ramp-duration")]
    pub warmup_prefill_concurrency_ramp_duration: Option<f64>,

    /// Warmup concurrency-ramp duration (`--warmup-concurrency-ramp-duration`).
    #[arg(long = "warmup-concurrency-ramp-duration")]
    pub warmup_concurrency_ramp_duration: Option<f64>,

    /// Warmup rate-ramp duration (`--warmup-request-rate-ramp-duration`).
    #[arg(long = "warmup-request-rate-ramp-duration")]
    pub warmup_request_rate_ramp_duration: Option<f64>,

    /// Warmup duration bound, seconds (`--warmup-duration`).
    #[arg(long = "warmup-duration")]
    pub warmup_duration: Option<f64>,

    /// Warmup grace period, seconds (`--warmup-grace-period`).
    #[arg(long = "warmup-grace-period")]
    pub warmup_grace_period: Option<f64>,

    /// Disable GPU telemetry collection (`--no-gpu-telemetry`).
    #[arg(long = "no-gpu-telemetry", default_value_t = false)]
    pub no_gpu_telemetry: bool,

    /// Custom DCGM exporter URLs (`--gpu-telemetry`), repeatable.
    #[arg(long = "gpu-telemetry", num_args = 1..)]
    pub gpu_telemetry: Vec<String>,

    /// Disable server-metrics collection (`--no-server-metrics`).
    #[arg(long = "no-server-metrics", default_value_t = false)]
    pub no_server_metrics: bool,

    /// Server-metrics output formats (`--server-metrics-formats`).
    #[arg(long = "server-metrics-formats", num_args = 1..)]
    pub server_metrics_formats: Vec<String>,

    /// Synthetic input length mean (`--isl` / `--synthetic-input-tokens-mean`);
    /// comma ⇒ sweep.
    #[arg(long = "isl", visible_aliases = ["synthetic-input-tokens-mean", "prompt-input-tokens-mean"])]
    pub isl: Option<String>,

    /// Synthetic input length stddev (`--isl-stddev`).
    #[arg(long = "isl-stddev", visible_aliases = ["synthetic-input-tokens-stddev", "prompt-input-tokens-stddev"])]
    pub isl_stddev: Option<f64>,

    /// Synthetic output length mean (`--osl` / `--output-tokens-mean`); comma ⇒ sweep.
    #[arg(long = "osl", visible_aliases = ["output-tokens-mean", "prompt-output-tokens-mean"])]
    pub osl: Option<String>,

    /// Synthetic output length stddev (`--osl-stddev`).
    #[arg(long = "osl-stddev", visible_aliases = ["output-tokens-stddev", "prompt-output-tokens-stddev"])]
    pub osl_stddev: Option<f64>,

    /// Enable streaming responses (`--streaming`).
    #[arg(long = "streaming", default_value_t = false)]
    pub streaming: bool,

    /// Prompts per request (`--batch-size`).
    #[arg(long = "batch-size", visible_aliases = ["batch-size-text", "prompt-batch-size"], short = 'b')]
    pub batch_size: Option<u32>,

    /// Download video content from responses (`--download-video-content`).
    #[arg(long = "download-video-content", default_value_t = false)]
    pub download_video_content: bool,

    /// Extra request-body inputs `key:value` (`--extra-inputs`), repeatable.
    #[arg(long = "extra-inputs", num_args = 1..)]
    pub extra_inputs: Vec<String>,

    /// Custom server-metrics scrape URLs (`--server-metrics`), repeatable.
    #[arg(long = "server-metrics", num_args = 1..)]
    pub server_metrics: Vec<String>,

    /// Arrival distribution alias (`--arrival-pattern`): poisson/gamma/constant.
    #[arg(long = "arrival-pattern")]
    pub arrival_pattern: Option<String>,

    /// Gamma burstiness alias (`--vllm-burstiness`).
    #[arg(long = "vllm-burstiness")]
    pub vllm_burstiness: Option<f64>,

    /// Per-request timeout, seconds (`--request-timeout-seconds`).
    #[arg(long = "request-timeout-seconds")]
    pub request_timeout_seconds: Option<f64>,

    /// Emit legacy `max_tokens` (`--use-legacy-max-tokens`).
    #[arg(long = "use-legacy-max-tokens", default_value_t = false)]
    pub use_legacy_max_tokens: bool,

    /// Trust server-reported token counts (`--use-server-token-count`).
    #[arg(long = "use-server-token-count", default_value_t = false)]
    pub use_server_token_count: bool,

    /// Connection reuse policy (`--connection-reuse-strategy`):
    /// `pooled` (default), `never`, `sticky-user-sessions`.
    #[arg(long = "connection-reuse-strategy")]
    pub connection_reuse_strategy: Option<String>,

    /// Request body content type (`--request-content-type`):
    /// `application/json` or `multipart/form-data`.
    #[arg(long = "request-content-type")]
    pub request_content_type: Option<String>,

    /// Readiness-probe timeout, seconds (`--wait-for-model-timeout`).
    #[arg(long = "wait-for-model-timeout")]
    pub wait_for_model_timeout: Option<f64>,

    /// Readiness-probe mode (`--wait-for-model-mode`): `models`/`inference`/`both`.
    #[arg(long = "wait-for-model-mode")]
    pub wait_for_model_mode: Option<String>,

    /// Readiness-probe interval, seconds (`--wait-for-model-interval`).
    #[arg(long = "wait-for-model-interval")]
    pub wait_for_model_interval: Option<f64>,

    /// Apply the chat template when tokenizing (`--apply-chat-template`).
    #[arg(long = "apply-chat-template", default_value_t = false)]
    pub apply_chat_template: bool,

    /// Prefill concurrency (`--prefill-concurrency`); requires `--streaming`.
    #[arg(long = "prefill-concurrency")]
    pub prefill_concurrency: Option<u32>,

    /// Ramp prefill concurrency over N seconds (`--prefill-concurrency-ramp-duration`).
    #[arg(long = "prefill-concurrency-ramp-duration")]
    pub prefill_concurrency_ramp_duration: Option<f64>,

    /// API authentication key (`--api-key`).
    #[arg(long = "api-key")]
    pub api_key: Option<String>,

    /// Custom HTTP headers `Name:value` (`--header` / `-H`), repeatable.
    #[arg(long = "header", short = 'H')]
    pub headers: Vec<String>,

    /// Concurrent requests to maintain (`--concurrency`); comma-list ⇒ sweep.
    #[arg(long = "concurrency")]
    pub concurrency: Option<String>,

    /// Target request rate, requests/second (`--request-rate`); comma ⇒ sweep.
    #[arg(long = "request-rate")]
    pub request_rate: Option<String>,

    /// Arrival distribution for `--request-rate` (`--request-rate-mode`):
    /// `poisson` (default), `gamma`, or `constant`.
    #[arg(long = "request-rate-mode")]
    pub request_rate_mode: Option<String>,

    /// Gamma burstiness/smoothness shape (`--arrival-smoothness`).
    #[arg(long = "arrival-smoothness")]
    pub arrival_smoothness: Option<f64>,

    /// Per-user request rate for user-centric mode (`--user-centric-rate`).
    #[arg(long = "user-centric-rate")]
    pub user_centric_rate: Option<f64>,

    /// Number of users for user-centric mode (`--num-users`).
    #[arg(long = "num-users")]
    pub num_users: Option<u32>,

    /// Ramp concurrency over N seconds (`--concurrency-ramp-duration`).
    #[arg(long = "concurrency-ramp-duration")]
    pub concurrency_ramp_duration: Option<f64>,

    /// Ramp request rate over N seconds (`--request-rate-ramp-duration`).
    #[arg(long = "request-rate-ramp-duration")]
    pub request_rate_ramp_duration: Option<f64>,

    /// Fraction of requests to cancel (`--request-cancellation-rate`).
    #[arg(long = "request-cancellation-rate")]
    pub request_cancellation_rate: Option<f64>,

    /// Delay before cancellation, seconds (`--request-cancellation-delay`).
    #[arg(long = "request-cancellation-delay")]
    pub request_cancellation_delay: Option<f64>,

    /// Maximum requests to send (`--request-count`); comma-list ⇒ sweep.
    #[arg(long = "request-count", visible_alias = "num-requests")]
    pub request_count: Option<String>,

    /// Maximum benchmark runtime, seconds (`--benchmark-duration`); comma ⇒ sweep.
    #[arg(long = "benchmark-duration")]
    pub benchmark_duration: Option<String>,

    /// Grace period after duration ends, seconds (`--benchmark-grace-period`).
    #[arg(long = "benchmark-grace-period")]
    pub benchmark_grace_period: Option<f64>,

    /// Warmup request count (`--warmup-request-count` / `--num-warmup-requests`).
    #[arg(long = "warmup-request-count", visible_alias = "num-warmup-requests")]
    pub warmup_request_count: Option<u64>,

    /// Warmup concurrency (`--warmup-concurrency`); defaults to profiling value.
    #[arg(long = "warmup-concurrency")]
    pub warmup_concurrency: Option<u32>,

    /// Warmup request rate (`--warmup-request-rate`).
    #[arg(long = "warmup-request-rate")]
    pub warmup_request_rate: Option<f64>,

    /// Artifact output directory (`--artifact-dir`).
    #[arg(long = "artifact-dir", visible_alias = "output-artifact-dir")]
    pub artifact_dir: Option<PathBuf>,

    /// Pre-configured public dataset to download (`--public-dataset`).
    #[arg(long = "public-dataset")]
    pub public_dataset: Option<String>,

    /// Input trace/dataset file or directory (`--input-file`).
    #[arg(long = "input-file")]
    pub input_file: Option<PathBuf>,

    /// Custom dataset format for `--input-file` (`--custom-dataset-type`).
    #[arg(long = "custom-dataset-type")]
    pub custom_dataset_type: Option<String>,

    /// HuggingFace subset for `--public-dataset` (`--hf-subset`).
    #[arg(long = "hf-subset")]
    pub hf_subset: Option<String>,

    /// Dataset sampling strategy (`--dataset-sampling-strategy`).
    #[arg(long = "dataset-sampling-strategy")]
    pub dataset_sampling_strategy: Option<String>,

    /// Shared system prompt length (`--shared-system-prompt-length`).
    #[arg(long = "shared-system-prompt-length")]
    pub shared_system_prompt_length: Option<u32>,

    /// Per-user context prompt length (`--user-context-prompt-length`).
    #[arg(long = "user-context-prompt-length")]
    pub user_context_prompt_length: Option<u32>,

    /// Number of prefix prompts / pool size (`--num-prefix-prompts`).
    #[arg(long = "num-prefix-prompts", visible_aliases = ["prompt-prefix-pool-size", "prefix-prompt-pool-size"])]
    pub num_prefix_prompts: Option<u32>,

    /// Prefix prompt length (`--prefix-prompt-length`).
    #[arg(long = "prefix-prompt-length", visible_alias = "prompt-prefix-length")]
    pub prefix_prompt_length: Option<u32>,

    /// Cap on inter-turn delay, seconds (`--inter-turn-delay-cap-seconds`).
    #[arg(long = "inter-turn-delay-cap-seconds")]
    pub inter_turn_delay_cap_seconds: Option<f64>,

    /// Synthetic video audio channels (`--video-audio-num-channels`).
    #[arg(long = "video-audio-num-channels")]
    pub video_audio_num_channels: Option<u32>,
    /// Synthetic video audio bit depth (`--video-audio-depth`).
    #[arg(long = "video-audio-depth")]
    pub video_audio_depth: Option<u32>,
    /// Synthetic video audio sample rate, Hz (`--video-audio-sample-rate`).
    #[arg(long = "video-audio-sample-rate")]
    pub video_audio_sample_rate: Option<f64>,

    /// Replay requests by their timestamps (`--fixed-schedule`).
    #[arg(long = "fixed-schedule", default_value_t = false)]
    pub fixed_schedule: bool,

    /// Auto-normalize fixed-schedule timestamps (`--fixed-schedule-auto-offset`).
    #[arg(long = "fixed-schedule-auto-offset")]
    pub fixed_schedule_auto_offset: Option<bool>,

    /// Fixed-schedule start offset (`--fixed-schedule-start-offset`).
    #[arg(long = "fixed-schedule-start-offset")]
    pub fixed_schedule_start_offset: Option<i64>,

    /// Fixed-schedule end offset (`--fixed-schedule-end-offset`).
    #[arg(long = "fixed-schedule-end-offset")]
    pub fixed_schedule_end_offset: Option<i64>,

    /// Model-selection strategy (`--model-selection-strategy`):
    /// `round_robin`/`random`/`weighted`.
    #[arg(long = "model-selection-strategy")]
    pub model_selection_strategy: Option<String>,

    /// Timeslice window, seconds (`--slice-duration`).
    #[arg(long = "slice-duration")]
    pub slice_duration: Option<f64>,

    /// Synthetic input-token block size (`--synthetic-input-tokens-block-size`
    /// / `--isl-block-size`).
    #[arg(
        long = "synthetic-input-tokens-block-size",
        visible_aliases = ["isl-block-size", "prompt-input-tokens-block-size"]
    )]
    pub isl_block_size: Option<u32>,

    /// Bounded-memory metric retention (`--sketch-metrics`).
    #[arg(long = "sketch-metrics", default_value_t = false)]
    pub sketch_metrics: bool,

    /// Synthetic image width mean, pixels (`--image-width-mean`).
    #[arg(long = "image-width-mean")]
    pub image_width_mean: Option<f64>,
    /// Synthetic image width stddev (`--image-width-stddev`).
    #[arg(long = "image-width-stddev")]
    pub image_width_stddev: Option<f64>,
    /// Synthetic image height mean, pixels (`--image-height-mean`).
    #[arg(long = "image-height-mean")]
    pub image_height_mean: Option<f64>,
    /// Synthetic image height stddev (`--image-height-stddev`).
    #[arg(long = "image-height-stddev")]
    pub image_height_stddev: Option<f64>,
    /// Synthetic images per request (`--image-batch-size`).
    #[arg(long = "image-batch-size", visible_alias = "batch-size-image")]
    pub image_batch_size: Option<u32>,
    /// Synthetic image format (`--image-format`).
    #[arg(long = "image-format")]
    pub image_format: Option<String>,
    /// Synthetic image source (`--image-source`).
    #[arg(long = "image-source")]
    pub image_source: Option<String>,
    /// Synthetic image source sampling (`--image-source-sampling`).
    #[arg(long = "image-source-sampling")]
    pub image_source_sampling: Option<String>,

    /// Synthetic audio length mean, seconds (`--audio-length-mean`).
    #[arg(long = "audio-length-mean")]
    pub audio_length_mean: Option<f64>,
    /// Synthetic audio length stddev (`--audio-length-stddev`).
    #[arg(long = "audio-length-stddev")]
    pub audio_length_stddev: Option<f64>,
    /// Synthetic audio clips per request (`--audio-batch-size`).
    #[arg(long = "audio-batch-size", visible_alias = "batch-size-audio")]
    pub audio_batch_size: Option<u32>,
    /// Synthetic audio channels (`--audio-num-channels`).
    #[arg(long = "audio-num-channels")]
    pub audio_num_channels: Option<u32>,
    /// Synthetic audio bit depths (`--audio-depths`).
    #[arg(long = "audio-depths", num_args = 1..)]
    pub audio_depths: Vec<u32>,
    /// Synthetic audio format (`--audio-format`).
    #[arg(long = "audio-format")]
    pub audio_format: Option<String>,
    /// Synthetic audio sample rates, Hz (`--audio-sample-rates`).
    #[arg(long = "audio-sample-rates", num_args = 1..)]
    pub audio_sample_rates: Vec<f64>,

    /// Synthetic video width, pixels (`--video-width`).
    #[arg(long = "video-width")]
    pub video_width: Option<u32>,
    /// Synthetic video height, pixels (`--video-height`).
    #[arg(long = "video-height")]
    pub video_height: Option<u32>,
    /// Synthetic video duration, seconds (`--video-duration`).
    #[arg(long = "video-duration")]
    pub video_duration: Option<f64>,
    /// Synthetic video frames per second (`--video-fps`).
    #[arg(long = "video-fps")]
    pub video_fps: Option<u32>,
    /// Synthetic video container format (`--video-format`).
    #[arg(long = "video-format")]
    pub video_format: Option<String>,
    /// Synthetic video codec (`--video-codec`).
    #[arg(long = "video-codec")]
    pub video_codec: Option<String>,

    /// Synthetic video AUDIO codec (`--video-audio-codec`; aac/libvorbis/libopus).
    #[arg(long = "video-audio-codec")]
    pub video_audio_codec: Option<String>,
    /// Synthetic video synthesis pattern (`--video-synth-type`).
    #[arg(long = "video-synth-type")]
    pub video_synth_type: Option<String>,
    /// Synthetic video clips per request (`--video-batch-size`).
    #[arg(long = "video-batch-size", visible_alias = "batch-size-video")]
    pub video_batch_size: Option<u32>,

    /// Enable adaptive scaling (`--adaptive-scale`).
    #[arg(long = "adaptive-scale", default_value_t = false)]
    pub adaptive_scale: bool,
    /// Adaptive control variable (`--adaptive-control-variable`).
    #[arg(long = "adaptive-control-variable")]
    pub adaptive_control_variable: Option<String>,
    /// Adaptive control minimum (`--adaptive-control-min`).
    #[arg(long = "adaptive-control-min")]
    pub adaptive_control_min: Option<i64>,
    /// Adaptive control maximum (`--adaptive-control-max`).
    #[arg(long = "adaptive-control-max")]
    pub adaptive_control_max: Option<i64>,
    /// Adaptive assessment period, seconds (`--adaptive-assessment-period`).
    #[arg(
        long = "adaptive-assessment-period",
        visible_alias = "adaptive-scale-assessment-period"
    )]
    pub adaptive_assessment_period: Option<f64>,
    /// Adaptive sustain duration, seconds (`--adaptive-sustain-duration`).
    #[arg(long = "adaptive-sustain-duration")]
    pub adaptive_sustain_duration: Option<f64>,
    /// Adaptive SLA filters `metric:stat:op:threshold` (`--adaptive-scale-sla`).
    #[arg(long = "adaptive-scale-sla", num_args = 1..)]
    pub adaptive_scale_sla: Vec<String>,

    /// Deterministic random seed (`--random-seed`).
    #[arg(long = "random-seed")]
    pub random_seed: Option<u64>,

    /// Goodput SLO thresholds, `metric:ms` space-separated (`--goodput`).
    #[arg(long = "goodput")]
    pub goodput: Option<String>,

    /// Fixed mean network RTT, milliseconds (`--network-latency-mean`).
    #[arg(long = "network-latency-mean")]
    pub network_latency_mean: Option<f64>,

    /// Enable automatic RTT calibration (`--network-latency-automatic`).
    #[arg(long = "network-latency-automatic", default_value_t = false)]
    pub network_latency_automatic: bool,

    /// RTT probe ping interval, seconds (`--network-latency-ping-interval`).
    #[arg(long = "network-latency-ping-interval")]
    pub network_latency_ping_interval: Option<f64>,

    /// OTLP/HTTP metrics collector URL (`--otel-url`).
    #[arg(long = "otel-url")]
    pub otel_url: Option<String>,
    /// GenAI provider label attached to OTLP metrics (`--gen-ai-provider`).
    #[arg(long = "gen-ai-provider")]
    pub gen_ai_provider: Option<String>,
    /// Extra OTLP resource attributes (`--otel-resource-attributes k=v`), repeatable.
    #[arg(long = "otel-resource-attributes", num_args = 1..)]
    pub otel_resource_attributes: Vec<String>,

    /// MLflow tracking server URI (`--mlflow-tracking-uri`).
    #[arg(long = "mlflow-tracking-uri")]
    pub mlflow_tracking_uri: Option<String>,
    /// MLflow experiment name (`--mlflow-experiment`).
    #[arg(long = "mlflow-experiment")]
    pub mlflow_experiment: Option<String>,
    /// MLflow run name (`--mlflow-run-name`).
    #[arg(long = "mlflow-run-name")]
    pub mlflow_run_name: Option<String>,
    /// Parent MLflow run id (`--mlflow-parent-run-id`).
    #[arg(long = "mlflow-parent-run-id")]
    pub mlflow_parent_run_id: Option<String>,
    /// MLflow run tags (`--mlflow-tag k:v`), repeatable.
    #[arg(long = "mlflow-tag", num_args = 1..)]
    pub mlflow_tag: Vec<String>,
    /// MLflow artifact glob override (`--mlflow-artifact-glob`), repeatable.
    #[arg(long = "mlflow-artifact-glob", num_args = 1..)]
    pub mlflow_artifact_glob: Vec<String>,

    /// W&B project (`--wandb-project`).
    #[arg(long = "wandb-project")]
    pub wandb_project: Option<String>,
    /// W&B entity (`--wandb-entity`).
    #[arg(long = "wandb-entity")]
    pub wandb_entity: Option<String>,
    /// W&B run name (`--wandb-run-name`).
    #[arg(long = "wandb-run-name")]
    pub wandb_run_name: Option<String>,
    /// W&B run tags (`--wandb-tag`), repeatable.
    #[arg(long = "wandb-tag", num_args = 1..)]
    pub wandb_tag: Vec<String>,

    /// YAML configuration file (`--config` / `-f`).
    #[arg(long = "config", short = 'f')]
    pub config_file: Option<PathBuf>,
}

impl ProfileFlags {
    /// Parse `profile` flags from an argv slice (program name already stripped,
    /// i.e. the tokens after `aiperf profile`).
    pub fn parse_from_args(args: &[String]) -> Result<Self, clap::Error> {
        // clap expects argv[0] to be the binary name.
        let mut argv = vec!["profile".to_string()];
        argv.extend_from_slice(args);
        Self::try_parse_from(argv)
    }
}
