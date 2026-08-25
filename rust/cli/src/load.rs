// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Parse `profile` flags or YAML config into a [`BenchmarkRun`].
//!
//! Flag and YAML inputs normalize to `Inputs` and share the resolution, keeping
//! wire defaults in one place. The `Inputs` -> `BenchmarkRun` resolution now
//! lives in `aiperf_runtime::config::resolve`; this module owns only the flag
//! authoring path (flags -> `Inputs`) and re-exports the moved types so its call
//! sites (and the YAML authoring path) resolve unchanged.

use std::path::PathBuf;
use std::str::FromStr;

// The normalized-inputs types and the resolution moved into the runtime; re-export
// them so `crate::load::{Inputs, Warmup, DatasetAnalysisInputs}` and `load::build`
// call sites in the flag and YAML authoring paths resolve unchanged.
pub use aiperf_runtime::config::resolve::{
    DatasetAnalysisInputs, Inputs, Warmup, resolve as build,
};

use crate::flags::ProfileFlags;
use crate::model::BenchmarkRun;
use crate::model::dataset::{
    AudioSpec, Distribution, ImageSpec, PrefixPrompts, RecordedAgentGraphConfig,
    RecordedAgentSourceFormat, VideoAudio, VideoSpec,
};
use crate::model::endpoint::{
    ConnectionReuse, RequestContentType, ResetKvCacheConfig, ServerProfilerConfig, WaitForModelMode,
};
use crate::model::models::ModelStrategy;
use crate::model::phase::{AdaptiveScale, SlaFilter};
use crate::model::rate_series::RateSeries;
use crate::model::transport::Transport;

const DEFAULT_ISL_MEAN: f64 = 550.0;
/// Default synthetic conversation count when no request bound is given.
pub(crate) const DEFAULT_ENTRIES: u32 = 100;

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
    (flags.reset_kv_cache.unwrap_or(false)
        || flags.reset_kv_cache_timeout_seconds.is_some()
        || flags.reset_kv_cache_path.is_some())
    .then(|| ResetKvCacheConfig {
        timeout_seconds: flags.reset_kv_cache_timeout_seconds,
        path: flags.reset_kv_cache_path.clone(),
    })
}

pub(crate) fn server_profiler_from_flags(flags: &ProfileFlags) -> Option<ServerProfilerConfig> {
    (flags.server_profiler.unwrap_or(false)
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
    build(resolve_inputs(flags)?)
}

fn reject_inapplicable_recorded_agent_flags(flags: &ProfileFlags) -> anyhow::Result<()> {
    if flags.graph_format.as_deref() == Some("agent_recording") {
        return Ok(());
    }

    for (is_authored, option) in [
        (
            flags.graph_recording_source.is_some(),
            "--graph-recording-source",
        ),
        (
            flags.graph_include_subagents.is_some(),
            "--graph-include-subagents",
        ),
        (flags.graph_replay_root.is_some(), "--graph-replay-root"),
        (flags.graph_execute_tools.is_some(), "--graph-execute-tools"),
        (flags.graph_tool_image.is_some(), "--graph-tool-image"),
        (flags.graph_pinch_image.is_some(), "--graph-pinch-image"),
        (
            flags.graph_tool_command_timeout.is_some(),
            "--graph-tool-command-timeout",
        ),
        (
            flags.graph_tool_container_stop_timeout.is_some(),
            "--graph-tool-container-stop-timeout",
        ),
        (
            flags.graph_tool_session_close_grace.is_some(),
            "--graph-tool-session-close-grace",
        ),
        (
            flags.graph_use_family_sampling.is_some(),
            "--graph-use-family-sampling",
        ),
        (
            flags.no_graph_use_family_sampling.is_some(),
            "--no-graph-use-family-sampling",
        ),
        (flags.graph_emit_warmup.is_some(), "--graph-emit-warmup"),
        (flags.graph_resume.is_some(), "--graph-resume"),
        (
            flags.graph_stop_on_failure.is_some(),
            "--graph-stop-on-failure",
        ),
    ] {
        if is_authored {
            anyhow::bail!("{option} requires --graph-format agent_recording");
        }
    }
    Ok(())
}

/// Normalize profile flags into authoring [`Inputs`] without resolving them.
///
/// The single-run path serializes these authoring inputs onto the `--execute` wire
/// so the runtime performs the authoritative resolution; [`resolve`] additionally
/// builds the run for callers (sweeps, searches) that resolve CLI-side.
pub fn resolve_inputs(flags: &ProfileFlags) -> anyhow::Result<Inputs> {
    reject_inapplicable_recorded_agent_flags(flags)?;
    reject_sweep("--concurrency", flags.concurrency.as_deref())?;
    reject_sweep("--request-count", flags.request_count.as_deref())?;
    reject_sweep("--request-rate", flags.request_rate.as_deref())?;
    reject_sweep("--benchmark-duration", flags.benchmark_duration.as_deref())?;
    reject_sweep("--num-conversations", flags.num_conversations.as_deref())?;

    if flags.use_think_time_only.unwrap_or(false) && flags.ignore_trace_delays.unwrap_or(false) {
        anyhow::bail!("--use-think-time-only and --ignore-trace-delays are mutually exclusive");
    }

    anyhow::ensure!(
        !flags.model_names.is_empty(),
        "at least one --model is required"
    );
    // A dry run opens no sockets, so a real endpoint URL is not required — default
    // a sentinel so the endpoint/profile lowering (which still wants some URL) is
    // satisfied. The fake transport never dials it.
    let urls = if flags.urls.is_empty() && flags.dry_run.unwrap_or(false) {
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
    let show_trace_timing = flags.show_trace_timing.unwrap_or(false);
    let export_trace = flags.export_http_trace.unwrap_or(false) || show_trace_timing;

    // `--hf-weka-dataset` auto-selects public dataset `weka_hf`.
    let hf_weka_dataset = flags.hf_weka_dataset.clone();
    let public_dataset = match (&flags.public_dataset, &hf_weka_dataset) {
        (Some(name), Some(_)) if name != "weka_hf" => {
            anyhow::bail!(
                "--hf-weka-dataset cannot be combined with --public-dataset {name}; omit --public-dataset or set it to weka_hf"
            );
        }
        (_, Some(_)) => Some("weka_hf".to_string()),
        (other, None) => other.clone(),
    };

    let allow_dataset_wrap = if flags.allow_dataset_wrap.unwrap_or(false) {
        Some(true)
    } else if flags.no_allow_dataset_wrap.unwrap_or(false) {
        Some(false)
    } else {
        None
    };
    let cache_bust = flags.cache_bust.clone().filter(|t| t != "none");
    let graph_recording_source = flags
        .graph_recording_source
        .as_deref()
        .map(RecordedAgentSourceFormat::from_str)
        .transpose()?
        .unwrap_or_default();
    let recorded_agent_graph =
        (flags.graph_format.as_deref() == Some("agent_recording")).then(|| {
            RecordedAgentGraphConfig {
                source_format: graph_recording_source,
                include_subagents: flags.graph_include_subagents,
                replay_root: flags.graph_replay_root.clone(),
                execute_tools: flags.graph_execute_tools.unwrap_or(false),
                tool_image: flags.graph_tool_image.clone(),
                pinch_image: flags.graph_pinch_image.clone(),
                command_timeout_seconds: flags.graph_tool_command_timeout.unwrap_or(900.0),
                container_stop_timeout_seconds: flags
                    .graph_tool_container_stop_timeout
                    .unwrap_or(5.0),
                session_close_grace_seconds: flags.graph_tool_session_close_grace.unwrap_or(1.0),
                use_family_sampling: flags.graph_use_family_sampling.unwrap_or(true)
                    && !flags.no_graph_use_family_sampling.unwrap_or(false),
                emit_warmup: flags.graph_emit_warmup.unwrap_or(false),
                resume: flags.graph_resume.unwrap_or(false),
                stop_on_failure: flags.graph_stop_on_failure.unwrap_or(false),
            }
        });

    // Fixed-schedule replays each timestamped entry once, so the request bound is
    // the schedule length (the input file's non-empty line count).
    // `--no-fixed-schedule` wins over `--fixed-schedule` (clap overrides_with).
    let want_fixed =
        flags.fixed_schedule.unwrap_or(false) && !flags.no_fixed_schedule.unwrap_or(false);
    let (fixed_schedule, request_count) = if want_fixed {
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
        transport: if flags.dry_run.unwrap_or(false) {
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
                virtual_workers: None,
            })
        } else {
            Transport::Http
        },
        streaming: flags.streaming.unwrap_or(false),
        timeout_seconds: flags.request_timeout_seconds,
        use_legacy_max_tokens: flags.use_legacy_max_tokens.unwrap_or(false),
        use_server_token_count: flags.use_server_token_count.unwrap_or(false),
        download_video_content: flags.download_video_content.unwrap_or(false),
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
        ssl_verify: flags.ssl_verify,
        uds_path: flags.uds_path.clone(),
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
        apply_chat_template: flags.apply_chat_template.unwrap_or(false),
        prefill_concurrency: flags.prefill_concurrency,
        prefill_ramp: flags.prefill_concurrency_ramp_duration,
        gpu_telemetry_enabled: !flags.no_gpu_telemetry.unwrap_or(false),
        // No flag selects a collector; YAML is the only authoring surface for it.
        gpu_telemetry_collector: None,
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
        server_metrics_enabled: !flags.no_server_metrics.unwrap_or(false),
        server_metrics_formats: (!flags.server_metrics_formats.is_empty())
            .then(|| flags.server_metrics_formats.clone()),
        slos: parse_goodput(flags.goodput.as_deref())?,
        network_latency_mean: flags.network_latency_mean,
        network_latency_probe: flags
            .network_latency_automatic
            .unwrap_or(false)
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
            sync_url: flags.wandb_sync_url.clone(),
        },
        api_key: flags.api_key.clone(),
        headers: parse_headers(&flags.headers)?,
        tokenizer_name: flags.tokenizer.clone(),
        tokenizer_revision: flags.tokenizer_revision.clone(),
        tokenizer_trust: flags.tokenizer_trust_remote_code.unwrap_or(false),
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
        proxy_from_env: flags.proxy_from_env.unwrap_or(false),
        endpoint_path: flags.custom_endpoint.clone(),
        reset_kv_cache: reset_kv_cache_from_flags(flags),
        server_profiler: server_profiler_from_flags(flags),
        records_formats,
        // No flag narrows the summary artifacts; the flag path ships both formats.
        summary_formats: Vec::new(),
        // User files are authorable only from Config v2; no flag renders them.
        user_files: Vec::new(),
        export_raw,
        export_trace,
        export_outputs_json: flags.export_outputs_json.unwrap_or(false),
        show_trace_timing,
        profile_export_prefix: flags.profile_export_prefix.clone(),
        use_think_time_only: flags.use_think_time_only.unwrap_or(false),
        max_context_length: flags.max_context_length,
        allow_dataset_wrap,
        cache_bust,
        burst_phase_starts: flags.burst_phase_starts.unwrap_or(false),
        trace_idle_gap_cap_seconds: flags.trace_idle_gap_cap_seconds,
        system_idle_gap_cap_seconds: flags.system_idle_gap_cap_seconds,
        hf_weka_dataset,
        trace_session_sample_ratio: flags.trace_session_sample_ratio,
        agentic_warmup_grace_period: flags.agentic_warmup_grace_period,
        failed_request_threshold: flags.failed_request_threshold,
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
        runtime_hop_routing: flags.hop_routing()?,
        input_file: flags.input_file.clone(),
        recorded_agent_graph,
        hardware_description: flags.hardware_description.clone(),
        endpoint_placement: flags
            .endpoint_placement
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        inline_records: None,
        custom_dataset_type: flags
            .graph_format
            .clone()
            .or_else(|| flags.custom_dataset_type.clone()),
        public_dataset,
        hf_subset: flags.hf_subset.clone(),
        hf_dataset: flags.hf_dataset.clone(),
        hf_split: flags.hf_split.clone(),
        hf_revision: flags.hf_revision.clone(),
        hf_text_column: flags.hf_text_column.clone(),
        hf_output_column: flags.hf_output_column.clone(),
        hf_output_len: flags.hf_output_len,
        hf_format: flags.hf_format.clone(),
        inter_turn_delay_cap_seconds: flags.inter_turn_delay_cap_seconds,
        prefetch_media_urls: flags.prefetch_media_urls.unwrap_or(false),
        uuid_and_strip: flags.uuid_and_strip.unwrap_or(false),
        replay_speedup: flags.replay_speedup,
        max_idle_gap_cap_seconds: flags.max_idle_gap_cap_seconds,
        open_loop_replay: flags.open_loop_replay.unwrap_or(true)
            && !flags.no_open_loop_replay.unwrap_or(false),
        open_loop_strict: flags.open_loop_strict.unwrap_or(false),
        omit_kv_hints: flags.omit_kv_hints.unwrap_or(false),
        force_min_tokens: flags.force_min_tokens.unwrap_or(true)
            && !flags.no_force_min_tokens.unwrap_or(false),
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
        sketch_metrics: flags.sketch_metrics.unwrap_or(false),
        steady_state: flags.steady_state.unwrap_or(false),
        steady_state_fraction: flags.steady_state_fraction,
        steady_state_hybrid: flags.steady_state_hybrid.unwrap_or(false),
        random_pool_image_batch_size: flags.image_batch_size,
        image_spec: build_image_spec(flags),
        audio_spec: build_audio_spec(flags),
        video_spec: build_video_spec(flags),
        adaptive_scale: build_adaptive_scale(flags, concurrency)?,
        prefix_prompts: build_prefix_prompts(flags),
        scenario: flags.scenario.clone(),
        weka_semantics: flags.weka_semantics.clone(),
        ignore_trace_delays: flags.ignore_trace_delays.unwrap_or(false),
        ignore_trace_delays_explicit: flags.ignore_trace_delays.is_some(),
        trajectory_start_min_ratio: flags.trajectory_start_min_ratio.unwrap_or(0.0),
        trajectory_start_max_ratio: flags.trajectory_start_max_ratio.unwrap_or(0.0),
        unsafe_override: flags.unsafe_override.unwrap_or(false),
        agentic_cache_warmup_duration: flags.agentic_cache_warmup_duration,
        rankings: build_rankings(flags),
        accuracy: build_accuracy(flags),
        synthesis: build_synthesis(flags)?,
        dataset_filters: parse_dataset_filters(flags)?,
        // Dry-run emits the dataset-analysis artifact family unless suppressed.
        dataset_analysis: (flags.dry_run.unwrap_or(false)
            && !flags.no_dataset_analysis.unwrap_or(false))
        .then(|| DatasetAnalysisInputs {
            block_size: flags.kv_block_size,
            cache_blocks: flags.kv_cache_blocks,
            per_conversation: flags.dataset_analysis_per_conversation.unwrap_or(false),
        }),
        phases_override: None,
        artifact_dir: flags
            .artifact_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("artifacts")),
    };
    Ok(inputs)
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

/// Parse repeatable `--extra-inputs` entries into a typed JSON map.
///
/// Each entry is either a `key:value` pair or a whole JSON object. The JSON
/// form is the only way to express a nested value (`{"stream_options":
/// {"include_usage": true}}`): `key:value` splits on the first colon, so a
/// nested object would otherwise be shredded into a garbage key and an
/// unparsable request body.
fn parse_extra_inputs(
    raw: &[String],
) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    let mut extra = serde_json::Map::new();
    for entry in raw {
        let trimmed = entry.trim();
        if trimmed.starts_with('{') {
            let parsed: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
                anyhow::anyhow!("invalid --extra-inputs JSON object {entry:?}: {e}")
            })?;
            let serde_json::Value::Object(map) = parsed else {
                anyhow::bail!("invalid --extra-inputs {entry:?}; JSON must be an object");
            };
            extra.extend(map);
            continue;
        }
        let (key, value) = trimmed.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("invalid --extra-inputs {entry:?}; expected key:value or a JSON object")
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
    if !flags.adaptive_scale.unwrap_or(false) {
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
pub(crate) fn build_synthesis(flags: &ProfileFlags) -> anyhow::Result<Option<serde_json::Value>> {
    let allow_wrap = if flags.allow_dataset_wrap.unwrap_or(false) {
        Some(true)
    } else if flags.no_allow_dataset_wrap.unwrap_or(false) {
        Some(false)
    } else {
        None
    };
    let cache_bust = flags.cache_bust.clone().filter(|t| t != "none");
    let any = flags.synthesis_speedup_ratio.is_some()
        || flags.synthesis_prefix_len_multiplier.is_some()
        || flags.synthesis_prefix_root_multiplier.is_some()
        || flags.synthesis_prompt_len_multiplier.is_some()
        || flags.synthesis_output_len_multiplier.is_some()
        || flags.synthesis_max_isl.is_some()
        || flags.synthesis_max_osl.is_some()
        || flags.synthesis_idle_gap_cap.is_some()
        || flags.trace_idle_gap_cap_seconds.is_some()
        || flags.max_context_length.is_some()
        || allow_wrap.is_some()
        || cache_bust.is_some();
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
    // Prefer `--trace-idle-gap-cap-seconds` over `--synthesis-idle-gap-cap`.
    let idle_gap = flags
        .trace_idle_gap_cap_seconds
        .or(flags.synthesis_idle_gap_cap)
        .unwrap_or(60.0);
    m.insert("idle_gap_cap_seconds".into(), f(idle_gap)?);
    if let Some(v) = flags.max_context_length {
        m.insert("max_context_length".into(), serde_json::Value::from(v));
    }
    if let Some(wrap) = allow_wrap {
        m.insert("allow_dataset_wrap".into(), serde_json::Value::Bool(wrap));
    }
    if let Some(target) = cache_bust {
        m.insert(
            "cache_bust_target".into(),
            serde_json::Value::String(target),
        );
    }
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
    let enable_cot = if flags.accuracy_enable_cot.unwrap_or(false) {
        Some(true)
    } else if flags.accuracy_no_enable_cot.unwrap_or(false) {
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
        verbose: flags.accuracy_verbose.unwrap_or(false),
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
    #[test]
    fn parses_system_idle_gap_cap_seconds() {
        run_on_big_stack(|| {
            use crate::flags::ProfileFlags;

            let flags = ProfileFlags::parse_from_args(&[
                "--system-idle-gap-cap-seconds".to_string(),
                "10".to_string(),
            ])
            .expect("parse flags");

            assert_eq!(flags.system_idle_gap_cap_seconds, Some(10.0));
        });
    }

    #[test]
    fn dry_run_projects_idle_gap_caps_without_an_endpoint() {
        run_on_big_stack(|| {
            use crate::flags::ProfileFlags;

            let flags = ProfileFlags::parse_from_args(&[
                "--dry-run".to_string(),
                "--model".to_string(),
                "test-model".to_string(),
                "--endpoint-type".to_string(),
                "chat".to_string(),
                "--trace-idle-gap-cap-seconds".to_string(),
                "12".to_string(),
                "--system-idle-gap-cap-seconds".to_string(),
                "7".to_string(),
            ])
            .expect("parse flags");
            let inputs = super::resolve_inputs(&flags).expect("resolve dry-run inputs");

            assert_eq!(inputs.trace_idle_gap_cap_seconds, Some(12.0));
            assert_eq!(inputs.system_idle_gap_cap_seconds, Some(7.0));
            assert!(matches!(
                inputs.transport,
                crate::model::transport::Transport::DryRun(_)
            ));
        });
    }

    #[test]
    fn synthesis_rejects_non_finite_value() {
        run_on_big_stack(|| {
            use crate::flags::ProfileFlags;
            use clap::Parser;

            for bad in ["nan", "inf", "-inf"] {
                let flags = ProfileFlags::try_parse_from([
                    "profile",
                    &format!("--synthesis-speedup-ratio={bad}"),
                ])
                .expect("flags parse");
                let err = super::build_synthesis(&flags)
                    .expect_err("non-finite synthesis value must be a clean error, not a panic");
                assert!(
                    err.to_string().contains("finite"),
                    "expected a finiteness error for {bad:?}, got: {err}"
                );
            }
        });
    }

    #[test]
    fn synthesis_accepts_finite_values() {
        run_on_big_stack(|| {
            use crate::flags::ProfileFlags;
            use clap::Parser;

            let flags =
                ProfileFlags::try_parse_from(["profile", "--synthesis-speedup-ratio", "2.5"])
                    .expect("flags parse");
            let value = super::build_synthesis(&flags)
                .expect("finite value builds")
                .expect("synthesis flag set yields Some");
            assert_eq!(value["speedup_ratio"], serde_json::json!(2.5));
        });
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
        assert!(
            error.to_string().contains("--input-file")
                || error.to_string().contains("--hf-dataset")
                || error.to_string().contains("public dataset"),
            "got: {error}"
        );

        // Non-baseten public dataset.
        let flags = parse(&[
            "-m",
            "mock-model",
            "--endpoint-type",
            "chat",
            "-u",
            "http://localhost:8000",
            "--public-dataset",
            "sharegpt",
            "--replay-speedup",
            "2.0",
        ]);
        let error = super::resolve(&flags).unwrap_err();
        assert!(error.to_string().contains("baseten_trace loader"));
        assert!(error.to_string().contains("sharegpt"), "got: {error}");
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
    fn baseten_only_flags_accepted_with_hf_baseten_format() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--hf-dataset",
                "org/baseten-traces",
                "--hf-format",
                "baseten_trace",
                "--trace-session-sample-ratio",
                "0.25",
                "--replay-speedup",
                "2.0",
            ]);
            let run = super::resolve(&flags).expect("public/hf baseten must accept knobs");
            let public = match run.cfg.datasets.as_ref().and_then(|d| d.first()) {
                Some(crate::model::dataset::Dataset::Public(public)) => public,
                other => panic!("expected public dataset, got {other:?}"),
            };
            assert_eq!(public.format, "baseten_trace");
            assert_eq!(
                public.options.get("trace_session_sample_ratio"),
                Some(&serde_json::json!(0.25))
            );
            assert_eq!(
                public.options.get("replay_speedup"),
                Some(&serde_json::json!(2.0))
            );
        });
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
    fn image_batch_size_projects_file_random_pool_options() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--input-file",
                "image-pool.jsonl",
                "--custom-dataset-type",
                "random_pool",
                "--image-batch-size",
                "4",
            ]);
            let run = super::resolve(&flags).expect("resolve run");
            let value = serde_json::to_value(&run).expect("serialize run");
            assert_eq!(
                value["cfg"]["datasets"][0]["options"]["image_batch_size"],
                serde_json::json!(4)
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

    #[test]
    fn scenario_agentx_defaults_global_idle_guard_for_both_weka_semantics() {
        for semantics in ["legacy", "graph-ir"] {
            run_on_big_stack(move || {
                let flags = parse(&[
                    "-m",
                    "mock-model",
                    "--endpoint-type",
                    "chat",
                    "-u",
                    "http://localhost:8000",
                    "--streaming",
                    "--input-file",
                    "/tmp/agentx-global-idle-guard.jsonl",
                    "--custom-dataset-type",
                    "weka_trace",
                    "--scenario",
                    "inferencex-agentx-mvp",
                    "--weka-semantics",
                    semantics,
                ]);
                let run = super::resolve(&flags).expect("AgentX scenario resolves");
                assert_eq!(run.cfg.weka_semantics.as_deref(), Some(semantics));
                assert_eq!(run.cfg.system_idle_gap_cap_seconds, Some(10.0));

                let dataset = run
                    .cfg
                    .datasets
                    .as_ref()
                    .and_then(|datasets| datasets.first())
                    .expect("resolved Weka dataset");
                let dataset = serde_json::to_value(dataset).expect("serialize dataset");
                assert!(
                    dataset.pointer("/synthesis/idle_gap_cap_seconds").is_none(),
                    "AgentX must not synthesize a per-trace cap under {semantics}: {dataset}"
                );
                assert!(
                    dataset
                        .pointer("/options/inter_turn_delay_cap_seconds")
                        .is_none(),
                    "AgentX must not synthesize a per-turn cap under {semantics}: {dataset}"
                );
                assert_eq!(
                    run.resolved
                        .scenario_outcome
                        .as_ref()
                        .and_then(|outcome| outcome["submission_valid"].as_bool()),
                    Some(true)
                );
            });
        }
    }

    #[test]
    fn scenario_agentx_legacy_idle_caps_fail_or_record_unsafe_override() {
        for (flag, message) in [
            ("--trace-idle-gap-cap-seconds", "per-trace request timing"),
            ("--inter-turn-delay-cap-seconds", "per-turn delay cap"),
        ] {
            run_on_big_stack(move || {
                let base = [
                    "-m",
                    "mock-model",
                    "--endpoint-type",
                    "chat",
                    "-u",
                    "http://localhost:8000",
                    "--streaming",
                    "--input-file",
                    "/tmp/agentx-global-idle-guard.jsonl",
                    "--custom-dataset-type",
                    "weka_trace",
                    "--scenario",
                    "inferencex-agentx-mvp",
                    flag,
                    "10",
                ];
                let flags = parse(&base);
                let error = super::resolve(&flags)
                    .expect_err("legacy cap must violate the AgentX scenario");
                assert!(error.to_string().contains(message), "unexpected: {error}");

                let mut overridden = base.to_vec();
                overridden.push("--unsafe-override");
                let flags = parse(&overridden);
                let run = super::resolve(&flags)
                    .expect("unsafe override keeps the invalid run resolvable");
                let outcome = run
                    .resolved
                    .scenario_outcome
                    .as_ref()
                    .expect("scenario outcome");
                assert_eq!(outcome["submission_valid"], serde_json::json!(false));
                assert_eq!(
                    outcome["submission_invalid_reasons"],
                    serde_json::json!(["unsafe_override"])
                );
                assert_eq!(outcome["violations"][0]["flag"], flag);
            });
        }
    }

    /// `--agentic-cache-warmup-duration` on a non-weka run (no scenario, no
    /// `--weka-semantics`) is rejected: neither weka arm lowers the run, so the
    /// accelerated cache-warmup substage reaches no consumer and the flag is an
    /// invisible no-op.
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
            let err = super::resolve(&flags).expect_err("guard must reject non-weka run");
            assert!(
                err.to_string()
                    .contains("--agentic-cache-warmup-duration requires a weka reconstruction run"),
                "unexpected error: {err}"
            );
        });
    }

    /// Both weka arms consume the accelerated cache-warmup substage, so
    /// `--agentic-cache-warmup-duration` passes the guard under either
    /// `--weka-semantics` value (any failure must not be the guard's own
    /// rejection). `graph-ir` regressed once: the guard accepted only the legacy
    /// spelling while `build_pressure_recycle` consumed the value on both arms.
    #[test]
    fn agentic_cache_warmup_accepted_under_legacy_weka() {
        for semantics in ["legacy", "graph-ir"] {
            run_on_big_stack(move || {
                let flags = parse(&[
                    "-m",
                    "mock-model",
                    "--endpoint-type",
                    "chat",
                    "-u",
                    "http://localhost:8000",
                    "--streaming",
                    "--weka-semantics",
                    semantics,
                    "--agentic-cache-warmup-duration",
                    "5",
                ]);
                if let Err(err) = super::resolve(&flags) {
                    assert!(
                        !err.to_string().contains(
                            "--agentic-cache-warmup-duration requires a weka reconstruction run"
                        ),
                        "guard must not fire under --weka-semantics {semantics}: {err}"
                    );
                }
            });
        }
    }

    /// An unknown `--scenario` name is rejected during resolution.
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

    #[test]
    fn export_outputs_json_projects() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--export-outputs-json",
            ]);
            let inputs = super::resolve_inputs(&flags).expect("inputs");
            assert!(inputs.export_outputs_json);
            let run = super::resolve(&flags).expect("resolve");
            let arts = run.cfg.artifacts.expect("artifacts");
            assert_eq!(arts.outputs_path.as_deref(), Some("outputs.json"));
        });
    }

    #[test]
    fn hard_trio_flags_project() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--failed-request-threshold",
                "0.25",
                "--agentic-warmup-grace-period",
                "12.5",
            ]);
            let inputs = super::resolve_inputs(&flags).expect("inputs");
            assert_eq!(inputs.failed_request_threshold, Some(0.25));
            assert_eq!(inputs.agentic_warmup_grace_period, Some(12.5));
            let run = super::resolve(&flags).expect("resolve");
            let profiling = run
                .cfg
                .phases
                .as_ref()
                .expect("phases")
                .iter()
                .find(|p| !p.common.exclude_from_results)
                .expect("profiling");
            assert_eq!(profiling.common.failed_request_threshold, Some(0.25));
            assert_eq!(profiling.common.agentic_warmup_grace_period, Some(12.5));
        });
    }

    #[test]
    fn benchmark_duration_uses_default_profiling_grace() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--benchmark-duration",
                "5",
            ]);
            let run = super::resolve(&flags).expect("resolve");
            let profiling = run
                .cfg
                .phases
                .as_ref()
                .expect("phases")
                .iter()
                .find(|phase| !phase.common.exclude_from_results)
                .expect("profiling");
            assert_eq!(profiling.common.duration, Some(5.0));
            assert_eq!(profiling.common.grace_period, Some(30.0));
        });
    }

    #[test]
    fn explicit_benchmark_grace_period_overrides_default() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--benchmark-duration",
                "5",
                "--benchmark-grace-period",
                "0",
            ]);
            let run = super::resolve(&flags).expect("resolve");
            let profiling = run
                .cfg
                .phases
                .as_ref()
                .expect("phases")
                .iter()
                .find(|phase| !phase.common.exclude_from_results)
                .expect("profiling");
            assert_eq!(profiling.common.grace_period, Some(0.0));
        });
    }

    #[test]
    fn explicit_positive_benchmark_grace_period_is_preserved() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--benchmark-duration",
                "5",
                "--benchmark-grace-period",
                "15",
            ]);
            let run = super::resolve(&flags).expect("resolve");
            let profiling = run
                .cfg
                .phases
                .as_ref()
                .expect("phases")
                .iter()
                .find(|phase| !phase.common.exclude_from_results)
                .expect("profiling");
            assert_eq!(profiling.common.grace_period, Some(15.0));
        });
    }

    #[test]
    fn trace_session_sample_ratio_requires_baseten() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--trace-session-sample-ratio",
                "0.5",
            ]);
            let err = super::resolve(&flags).expect_err("baseten-only");
            assert!(err.to_string().contains("baseten_trace"), "got: {err}");
        });
    }

    #[test]
    fn use_think_time_only_mutex_with_ignore_trace_delays() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--use-think-time-only",
                "--ignore-trace-delays",
            ]);
            let err = super::resolve_inputs(&flags).expect_err("mutex");
            assert!(err.to_string().contains("mutually exclusive"), "{err}");
        });
    }

    #[test]
    fn hf_weka_dataset_auto_selects_public_weka_hf() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--hf-weka-dataset",
                "org/weka-traces",
            ]);
            let inputs = super::resolve_inputs(&flags).expect("inputs");
            assert_eq!(inputs.public_dataset.as_deref(), Some("weka_hf"));
            assert_eq!(inputs.hf_weka_dataset.as_deref(), Some("org/weka-traces"));
            let run = super::resolve(&flags).expect("resolve");
            let ds = &serde_json::to_value(&run).unwrap()["cfg"]["datasets"][0];
            assert_eq!(ds["type"], "public");
            assert_eq!(ds["name"], "weka_hf");
            assert_eq!(ds["format"], "weka_trace");
            assert_eq!(ds["source"]["dataset"], "org/weka-traces");
        });
    }

    #[test]
    fn allow_dataset_wrap_projects_into_synthesis() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--allow-dataset-wrap",
            ]);
            let inputs = super::resolve_inputs(&flags).expect("inputs");
            assert_eq!(inputs.allow_dataset_wrap, Some(true));
            let synth = inputs.synthesis.expect("synthesis");
            assert_eq!(synth["allow_dataset_wrap"], true);
        });
    }

    #[test]
    fn no_fixed_schedule_disables_fixed_schedule() {
        run_on_big_stack(|| {
            use std::io::Write;
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("sched.jsonl");
            let mut f = std::fs::File::create(&path).unwrap();
            writeln!(f, r#"{{"timestamp":0,"text":"a"}}"#).unwrap();
            writeln!(f, r#"{{"timestamp":1,"text":"b"}}"#).unwrap();
            let path_str = path.to_string_lossy().to_string();
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--input-file",
                &path_str,
                "--fixed-schedule",
                "--no-fixed-schedule",
            ]);
            let inputs = super::resolve_inputs(&flags).expect("inputs");
            assert!(inputs.fixed_schedule.is_none());
        });
    }

    #[test]
    fn profile_export_prefix_rewrites_artifact_stem() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--profile-export-prefix",
                "myrun",
            ]);
            let run = super::resolve(&flags).expect("resolve");
            let arts = run.cfg.artifacts.expect("artifacts");
            assert_eq!(arts.records_path.as_deref(), Some("myrun.jsonl"));
        });
    }

    #[test]
    fn profile_export_prefix_rewrites_gpu_telemetry_artifact_stem() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "-u",
                "http://localhost:8000",
                "--gpu-telemetry",
                "http://localhost:9400",
                "--profile-export-prefix",
                "myrun_gpu_telemetry.jsonl",
            ]);
            let run = super::resolve(&flags).expect("resolve");
            let arts = run.cfg.artifacts.expect("artifacts");
            assert_eq!(arts.records_path.as_deref(), Some("myrun.jsonl"));
            let sidecar = run
                .cfg
                .sidecars
                .and_then(|sidecars| sidecars.gpu_telemetry)
                .expect("GPU telemetry sidecar");
            assert_eq!(sidecar.records_path, "myrun_gpu_telemetry.jsonl");
        });
    }

    #[test]
    fn vary_seed_per_trial_offsets_trial_seed() {
        run_on_big_stack(|| {
            let flags = parse(&[
                "-m",
                "mock-model",
                "--endpoint-type",
                "chat",
                "--dry-run",
                "--random-seed",
                "10",
                "--vary-seed-per-trial",
            ]);
            let policy = crate::profile::seed_policy(&flags);
            assert_eq!(policy.seed(0, 0), Some(10));
            assert_eq!(policy.seed(0, 1), Some(11));
            assert_eq!(policy.seed(2, 1), Some(13));
        });
    }
}
