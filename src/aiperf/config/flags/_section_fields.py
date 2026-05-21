# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-section field-name sets for the ``CLIConfig``.

These frozensets group ``CLIConfig`` flat fields by their semantic section
(endpoint/input/output/tokenizer/loadgen/sweeping/accuracy). Resolvers and
converters use them to compute ``cli.model_fields_set & <SECTION>_FIELDS``
when they need "fields the user explicitly set in this section".

Disjointness across the seven sections is enforced by
``tests/unit/config/v1/test_section_fields.py``.
"""

from __future__ import annotations

ENDPOINT_FIELDS: frozenset[str] = frozenset(
    {
        "api_key",
        "connection_reuse_strategy",
        "custom_endpoint",
        "download_video_content",
        "model_names",
        "model_selection_strategy",
        "request_content_type",
        "session_header",
        "streaming",
        "timeout_seconds",
        "transport",
        "endpoint_type",
        "url_selection_strategy",
        "urls",
        "use_legacy_max_tokens",
        "use_server_token_count",
        "wait_for_model_interval",
        "wait_for_model_mode",
        "wait_for_model_timeout",
    }
)

INPUT_FIELDS: frozenset[str] = frozenset(
    {
        # ----- top-level input flat fields -----
        "custom_dataset_type",
        "dataset_sampling_strategy",
        "extra_inputs",
        "input_file",
        "fixed_schedule",
        "fixed_schedule_auto_offset",
        "fixed_schedule_end_offset",
        "fixed_schedule_start_offset",
        "goodput",
        "headers",
        "hf_dataset_subset",
        "public_dataset",
        "random_seed",
        # ----- conversation modality -----
        "conversation_num",
        "conversation_num_dataset_entries",
        "conversation_turn_mean",
        "conversation_turn_stddev",
        "conversation_turn_delay_mean",
        "conversation_turn_delay_stddev",
        "conversation_turn_delay_ratio",
        # ----- prompt modality -----
        "prompt_batch_size",
        "prompt_input_tokens_mean",
        "prompt_input_tokens_stddev",
        "prompt_input_tokens_block_size",
        "prompt_output_tokens_mean",
        "prompt_output_tokens_stddev",
        "prompt_prefix_pool_size",
        "prompt_prefix_length",
        "prompt_prefix_shared_system_length",
        "prompt_prefix_user_context_length",
        "prompt_sequence_distribution",
        # ----- image modality -----
        "image_width_mean",
        "image_width_stddev",
        "image_height_mean",
        "image_height_stddev",
        "image_batch_size",
        "image_format",
        "image_source",
        # ----- audio modality -----
        "audio_batch_size",
        "audio_length_mean",
        "audio_length_stddev",
        "audio_format",
        "audio_depths",
        "audio_sample_rates",
        "audio_num_channels",
        # ----- video modality -----
        "video_batch_size",
        "video_duration",
        "video_fps",
        "video_width",
        "video_height",
        "video_synth_type",
        "video_format",
        "video_codec",
        "video_audio_sample_rate",
        "video_audio_channels",
        "video_audio_codec",
        "video_audio_depth",
        # ----- rankings modality -----
        "rankings_passages_mean",
        "rankings_passages_stddev",
        "rankings_passages_prompt_token_mean",
        "rankings_passages_prompt_token_stddev",
        "rankings_query_prompt_token_mean",
        "rankings_query_prompt_token_stddev",
        # ----- synthesis modality -----
        "synthesis_speedup_ratio",
        "synthesis_prefix_len_multiplier",
        "synthesis_prefix_root_multiplier",
        "synthesis_prompt_len_multiplier",
        "synthesis_max_isl",
        "synthesis_max_osl",
    }
)

OUTPUT_FIELDS: frozenset[str] = frozenset(
    {
        "artifact_directory",
        "auto_plot",
        "export_http_trace",
        "export_level",
        "plot_required",
        "profile_export_prefix",
        "show_trace_timing",
        "slice_duration",
    }
)

TOKENIZER_FIELDS: frozenset[str] = frozenset(
    {
        "tokenizer_name",
        "tokenizer_revision",
        "trust_remote_code",
    }
)

LOADGEN_FIELDS: frozenset[str] = frozenset(
    {
        "arrival_pattern",
        "arrival_smoothness",
        "benchmark_duration",
        "benchmark_grace_period",
        "concurrency",
        "concurrency_ramp_duration",
        "num_users",
        "prefill_concurrency",
        "prefill_concurrency_ramp_duration",
        "request_cancellation_delay",
        "request_cancellation_rate",
        "request_count",
        "request_rate",
        "request_rate_ramp_duration",
        "user_centric_rate",
        "warmup_arrival_pattern",
        "warmup_concurrency",
        "warmup_concurrency_ramp_duration",
        "warmup_duration",
        "warmup_grace_period",
        "warmup_num_sessions",
        "warmup_prefill_concurrency",
        "warmup_prefill_concurrency_ramp_duration",
        "warmup_request_count",
        "warmup_request_rate",
        "warmup_request_rate_ramp_duration",
    }
)

SWEEPING_FIELDS: frozenset[str] = frozenset(
    {
        "bo_constraint_mode",
        "concurrency_max",
        "concurrency_min",
        "concurrency_steps",
        "confidence_level",
        "convergence_metric",
        "convergence_mode",
        "convergence_stat",
        "convergence_threshold",
        "degradation_metric_tag",
        "degradation_stat",
        "degradation_threshold",
        "e2e_sla_ms",
        "error_rate_sla",
        "isl_max",
        "isl_min",
        "isl_osl_pairs",
        "isl_steps",
        "itl_sla_ms",
        "no_sweep_table",
        "num_profile_runs",
        "optuna_acquisition",
        "optuna_sampler",
        "optuna_terminator",
        "osl_max",
        "osl_min",
        "osl_steps",
        "parameter_sweep_cooldown_seconds",
        "parameter_sweep_mode",
        "parameter_sweep_same_seed",
        "profile_run_cooldown_seconds",
        "profile_run_disable_warmup_after_first",
        "search_direction",
        "search_initial_points",
        "search_max_iterations",
        "search_metric",
        "search_percentile_pooling",
        "search_planner",
        "search_random_seed",
        "search_recipe",
        "search_sla",
        "search_space",
        "search_stat",
        "search_style",
        "set_consistent_seed",
        "slo_attainment_fraction",
        "tpot_sla_ms",
        "ttft_sla_ms",
        "sweep_variants",
        "vary_seed_per_trial",
    }
)

ACCURACY_FIELDS: frozenset[str] = frozenset(
    {
        "accuracy_benchmark",
        "accuracy_enable_cot",
        "accuracy_grader",
        "accuracy_n_shots",
        "accuracy_system_prompt",
        "accuracy_tasks",
        "accuracy_verbose",
    }
)
