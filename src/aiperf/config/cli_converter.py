# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI-to-config converter.

Converts a flat ``CLIModel`` into a nested ``AIPerfConfig`` dict, then
validates through Pydantic. No magic — just explicit field-by-field mapping.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

    from aiperf.config.config import AIPerfConfig


def build_aiperf_config(cli: BaseModel) -> AIPerfConfig:
    """Build an AIPerfConfig from a parsed CLIModel instance."""
    from aiperf.common.enums import (
        AIPerfLogLevel,
        CommunicationType,
        ExportFormat,
        ExportLevel,
    )
    from aiperf.common.metric_utils import normalize_metrics_endpoint_url
    from aiperf.common.utils import is_tty
    from aiperf.config.config import AIPerfConfig
    from aiperf.config.parsing import coerce_value
    from aiperf.config.phases import PhaseType
    from aiperf.plugin.enums import ArrivalPattern, EndpointType, UIType

    s = cli.model_fields_set

    def url(item: str) -> str:
        return item if item.startswith("http") else f"http://{item}"

    # =====================================================================
    # ENDPOINT
    # =====================================================================

    endpoint: dict[str, Any] = {"urls": [url(u) for u in cli.urls]}
    if "url_selection_strategy" in s:
        endpoint["url_strategy"] = cli.url_selection_strategy
    if "endpoint_type" in s:
        endpoint["type"] = cli.endpoint_type
    if "streaming" in s:
        endpoint["streaming"] = cli.streaming
    if "custom_endpoint" in s:
        endpoint["path"] = cli.custom_endpoint
    if "api_key" in s:
        endpoint["api_key"] = cli.api_key
    if "request_timeout_seconds" in s:
        endpoint["timeout"] = cli.request_timeout_seconds
    if "ready_check_timeout" in s:
        endpoint["ready_check_timeout"] = cli.ready_check_timeout
    if "transport_type" in s:
        endpoint["transport"] = cli.transport_type
    if "use_legacy_max_tokens" in s:
        endpoint["use_legacy_max_tokens"] = cli.use_legacy_max_tokens
    if "use_server_token_count" in s:
        endpoint["use_server_token_count"] = cli.use_server_token_count
    if "connection_reuse_strategy" in s:
        endpoint["connection_reuse"] = cli.connection_reuse_strategy
    if cli.headers:
        endpoint["headers"] = dict(cli.headers)
    if cli.extra_inputs:
        extra = dict(cli.extra_inputs)
        payload_template = extra.pop("payload_template", None)
        if payload_template is not None:
            path = Path(payload_template)
            body = path.read_text() if path.is_file() else payload_template
            endpoint["template"] = {
                "body": body,
                "response_field": extra.pop("response_field", "text"),
            }
        endpoint["extra"] = extra
    if endpoint.get("type") == EndpointType.TEMPLATE and "template" not in endpoint:
        extra_raw = endpoint.get("extra")
        if extra_raw:
            ex = dict(extra_raw) if isinstance(extra_raw, list) else extra_raw
            ts = ex.get("payload_template")
            if ts is not None:
                tp = Path(ts)
                endpoint["template"] = {"body": tp.read_text() if tp.is_file() else ts}

    # =====================================================================
    # MODELS
    # =====================================================================

    models: dict[str, Any] = {"items": [{"name": n} for n in cli.model_names]}
    if "model_selection_strategy" in s:
        models["strategy"] = cli.model_selection_strategy

    # =====================================================================
    # PROFILING PHASE
    # =====================================================================

    prof: dict[str, Any] = {}
    if "benchmark_duration" in s:
        prof["duration"] = cli.benchmark_duration
    if "benchmark_grace_period" in s:
        prof["grace_period"] = cli.benchmark_grace_period
    if "concurrency" in s:
        prof["concurrency"] = cli.concurrency
    if "prefill_concurrency" in s:
        prof["prefill_concurrency"] = cli.prefill_concurrency
    if "arrival_smoothness" in s:
        prof["smoothness"] = cli.arrival_smoothness
    if "request_count" in s:
        prof["requests"] = cli.request_count
    if "num_sessions" in s:
        prof["sessions"] = cli.num_sessions
    if "num_users" in s:
        prof["users"] = cli.num_users
    if "request_rate" in s:
        prof["rate"] = cli.request_rate
    if "user_centric_rate" in s:
        prof["rate"] = cli.user_centric_rate
    if "fixed_schedule_auto_offset" in s:
        prof["auto_offset"] = cli.fixed_schedule_auto_offset
    if "fixed_schedule_start_offset" in s:
        prof["start_offset"] = cli.fixed_schedule_start_offset
    if "fixed_schedule_end_offset" in s:
        prof["end_offset"] = cli.fixed_schedule_end_offset
    if "concurrency_ramp_duration" in s:
        prof["concurrency_ramp"] = {"duration": cli.concurrency_ramp_duration}
    if "prefill_concurrency_ramp_duration" in s:
        prof["prefill_ramp"] = {"duration": cli.prefill_concurrency_ramp_duration}
    if "request_rate_ramp_duration" in s:
        prof["rate_ramp"] = {"duration": cli.request_rate_ramp_duration}

    # phase type
    if cli.fixed_schedule:
        prof["type"] = PhaseType.FIXED_SCHEDULE
        if "start_offset" in prof:
            prof.setdefault("auto_offset", False)
    elif cli.user_centric_rate is not None:
        prof["type"] = PhaseType.USER_CENTRIC
    elif cli.request_rate is not None:
        match cli.arrival_pattern:
            case ArrivalPattern.GAMMA:
                prof["type"] = PhaseType.GAMMA
            case ArrivalPattern.CONSTANT:
                prof["type"] = PhaseType.CONSTANT
            case _:
                prof["type"] = PhaseType.POISSON
    else:
        prof["type"] = PhaseType.CONCURRENCY

    if (
        prof["type"] == PhaseType.USER_CENTRIC
        and (getattr(cli, "num_turns_mean", 1) or 1) < 2
    ):
        raise ValueError(
            "User-centric rate mode requires --session-turns-mean >= 2. "
            "For single-turn workloads, use --request-rate instead."
        )
    if (
        not any(k in prof for k in ("requests", "duration", "sessions"))
        and prof["type"] != PhaseType.FIXED_SCHEDULE
    ):
        prof.setdefault("requests", 10)
    if cli.request_cancellation_rate:
        cancel: dict[str, Any] = {"rate": cli.request_cancellation_rate}
        if cli.request_cancellation_delay is not None:
            cancel["delay"] = cli.request_cancellation_delay
        prof["cancellation"] = cancel

    # =====================================================================
    # WARMUP PHASE
    # =====================================================================

    phases: dict[str, Any] = {}

    if {"warmup_request_count", "warmup_num_sessions", "warmup_duration"} & s:
        w: dict[str, Any] = {"exclude_from_results": True}
        if cli.warmup_request_count is not None:
            w["requests"] = cli.warmup_request_count
        elif cli.warmup_num_sessions is not None:
            w["sessions"] = cli.warmup_num_sessions
        elif cli.warmup_duration is not None:
            w["duration"] = cli.warmup_duration

        warmup_rate = (
            cli.warmup_request_rate if "warmup_request_rate" in s else cli.request_rate
        )
        warmup_pattern = (
            cli.warmup_arrival_pattern
            if "warmup_arrival_pattern" in s
            else cli.arrival_pattern
        )
        warmup_concurrency = (
            cli.warmup_concurrency if "warmup_concurrency" in s else cli.concurrency
        ) or 1

        if warmup_rate is not None:
            w["rate"] = warmup_rate
            match warmup_pattern:
                case ArrivalPattern.GAMMA:
                    w["type"] = PhaseType.GAMMA
                    w["smoothness"] = cli.arrival_smoothness
                case ArrivalPattern.CONSTANT:
                    w["type"] = PhaseType.CONSTANT
                case _:
                    w["type"] = PhaseType.POISSON
        else:
            w["type"] = PhaseType.CONCURRENCY
        w["concurrency"] = warmup_concurrency

        # ramps: warmup-specific or fall back to profiling
        cr = (
            cli.warmup_concurrency_ramp_duration
            if "warmup_concurrency_ramp_duration" in s
            else (
                cli.concurrency_ramp_duration
                if "concurrency_ramp_duration" in s
                else None
            )
        )
        pr = (
            cli.warmup_prefill_concurrency_ramp_duration
            if "warmup_prefill_concurrency_ramp_duration" in s
            else (
                cli.prefill_concurrency_ramp_duration
                if "prefill_concurrency_ramp_duration" in s
                else None
            )
        )
        rr = (
            cli.warmup_request_rate_ramp_duration
            if "warmup_request_rate_ramp_duration" in s
            else (
                cli.request_rate_ramp_duration
                if "request_rate_ramp_duration" in s
                else None
            )
        )
        if cr is not None:
            w["concurrency_ramp"] = {"duration": cr}
        if pr is not None:
            w["prefill_ramp"] = {"duration": pr}
        if rr is not None:
            w["rate_ramp"] = {"duration": rr}

        if cli.warmup_prefill_concurrency is not None:
            w["prefill_concurrency"] = cli.warmup_prefill_concurrency
        if cli.warmup_grace_period is not None:
            w["grace_period"] = cli.warmup_grace_period

        phases["warmup"] = w

    phases["profiling"] = prof

    # =====================================================================
    # DATASET
    # =====================================================================

    ds = _build_dataset(cli, s)

    # =====================================================================
    # ARTIFACTS
    # =====================================================================

    from aiperf.common import random_generator as rng
    from aiperf.common.exceptions import InvalidStateError

    with contextlib.suppress(InvalidStateError):
        rng.init(cli.random_seed)

    args = [coerce_value(x) for x in sys.argv[1:]]
    cli_command = " ".join(
        ["aiperf"]
        + [
            f"'{x}'"
            if isinstance(x, str) and not x.startswith("-") and x != "profile"
            else str(x)
            for x in args
        ]
    )
    artifacts: dict[str, Any] = {"cli_command": cli_command}
    if "artifact_directory" in s:
        artifacts["dir"] = cli.artifact_directory
    if "slice_duration" in s:
        artifacts["slice_duration"] = cli.slice_duration
    if "export_http_trace" in s:
        artifacts["trace"] = cli.export_http_trace
    if "export_per_chunk_data" in s:
        artifacts["per_chunk_data"] = cli.export_per_chunk_data
    if "show_trace_timing" in s:
        artifacts["show_trace_timing"] = cli.show_trace_timing
    if cli.export_level in (ExportLevel.RECORDS, ExportLevel.RAW):
        artifacts["records"] = [ExportFormat.JSONL, "csv"]
    artifacts["raw"] = cli.export_level == ExportLevel.RAW
    if cli.profile_export_prefix:
        artifacts["prefix"] = Path(cli.profile_export_prefix).stem

    # =====================================================================
    # GPU TELEMETRY
    # =====================================================================

    if cli.no_gpu_telemetry:
        gpu_telemetry: dict[str, Any] = {"enabled": False}
    elif not cli.gpu_telemetry:
        gpu_telemetry = {"enabled": True}
    else:
        urls, metrics_file = [], None
        for item in cli.gpu_telemetry:
            if item.endswith(".csv"):
                metrics_file = Path(item)
            elif item.startswith("http") or ":" in item:
                urls.append(url(item))
        gpu_telemetry = {"enabled": True, "urls": urls}
        if metrics_file is not None:
            gpu_telemetry["metrics_file"] = metrics_file

    # =====================================================================
    # SERVER METRICS
    # =====================================================================

    if cli.no_server_metrics:
        server_metrics: dict[str, Any] = {"enabled": False}
    else:
        sm_urls = [
            normalize_metrics_endpoint_url(url(i))
            for i in cli.server_metrics or []
            if i.startswith("http") or ":" in i
        ]
        server_metrics = {"enabled": True, "urls": sm_urls}
        if cli.server_metrics_formats:
            server_metrics["formats"] = list(cli.server_metrics_formats)

    # =====================================================================
    # LOGGING + RUNTIME
    # =====================================================================

    logging_dict: dict[str, Any] = {}
    runtime_dict: dict[str, Any] = {}
    if "log_level" in s:
        logging_dict["level"] = cli.log_level
    if "ui_type" in s:
        runtime_dict["ui"] = cli.ui_type
    if "workers_max" in s:
        runtime_dict["workers"] = cli.workers_max
    if "record_processor_service_count" in s:
        runtime_dict["record_processors"] = cli.record_processor_service_count

    ui_set = "ui" in runtime_dict
    if cli.extra_verbose:
        logging_dict["level"] = AIPerfLogLevel.TRACE
        runtime_dict["ui"] = UIType.SIMPLE
    elif cli.verbose:
        logging_dict["level"] = AIPerfLogLevel.DEBUG
        runtime_dict["ui"] = UIType.SIMPLE
    elif not ui_set and not is_tty():
        runtime_dict["ui"] = UIType.NONE

    if cli.zmq_ipc_path is not None:
        runtime_dict["communication"] = {
            "type": CommunicationType.IPC,
            "path": str(cli.zmq_ipc_path),
        }
    elif cli.zmq_host is not None:
        runtime_dict["communication"] = {
            "type": CommunicationType.TCP,
            "host": cli.zmq_host,
        }

    # =====================================================================
    # ASSEMBLE
    # =====================================================================

    nested: dict[str, Any] = {
        "endpoint": endpoint,
        "models": models,
        "phases": phases,
        "datasets": {"main": ds},
        "artifacts": artifacts,
        "gpu_telemetry": gpu_telemetry,
        "server_metrics": server_metrics,
    }
    if logging_dict:
        nested["logging"] = logging_dict
    if runtime_dict:
        nested["runtime"] = runtime_dict

    # tokenizer
    tok: dict[str, Any] = {}
    if "tokenizer_name" in s:
        tok["name"] = cli.tokenizer_name
    if "tokenizer_revision" in s:
        tok["revision"] = cli.tokenizer_revision
    if "tokenizer_trust_remote_code" in s:
        tok["trust_remote_code"] = cli.tokenizer_trust_remote_code
    if tok:
        nested["tokenizer"] = tok

    # accuracy
    acc: dict[str, Any] = {}
    if "accuracy_benchmark" in s:
        acc["benchmark"] = cli.accuracy_benchmark
    if "accuracy_tasks" in s:
        acc["tasks"] = cli.accuracy_tasks
    if "accuracy_n_shots" in s:
        acc["n_shots"] = cli.accuracy_n_shots
    if "accuracy_enable_cot" in s:
        acc["enable_cot"] = cli.accuracy_enable_cot
    if "accuracy_grader" in s:
        acc["grader"] = cli.accuracy_grader
    if "accuracy_system_prompt" in s:
        acc["system_prompt"] = cli.accuracy_system_prompt
    if "accuracy_verbose" in s:
        acc["verbose"] = cli.accuracy_verbose
    if acc:
        nested["accuracy"] = acc

    # multi_run
    mr: dict[str, Any] = {}
    if "num_profile_runs" in s:
        mr["num_runs"] = cli.num_profile_runs
    if "profile_run_cooldown_seconds" in s:
        mr["cooldown_seconds"] = cli.profile_run_cooldown_seconds
    if "confidence_level" in s:
        mr["confidence_level"] = cli.confidence_level
    if "profile_run_disable_warmup_after_first" in s:
        mr["disable_warmup_after_first"] = cli.profile_run_disable_warmup_after_first
    if "set_consistent_seed" in s:
        mr["set_consistent_seed"] = cli.set_consistent_seed
    if mr:
        nested["multi_run"] = mr

    if "random_seed" in s:
        nested["random_seed"] = cli.random_seed
    if cli.goodput:
        nested["slos"] = dict(cli.goodput)

    return AIPerfConfig(**nested)


# =========================================================================
# DATASET (extracted — too many branches for inline)
# =========================================================================


def _build_dataset(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    """Build dataset dict."""
    from aiperf.common.enums import DatasetType
    from aiperf.plugin.plugins import get_endpoint_metadata

    # Validate non-tokenizing endpoint constraints
    endpoint_type = getattr(cli, "endpoint_type", None)
    needs_text = True
    if endpoint_type is not None:
        meta = get_endpoint_metadata(endpoint_type)
        needs_text = meta.tokenizes_input or meta.produces_tokens
        if not needs_text:
            _TEXT = {
                "isl_mean": (
                    "--isl",
                    "--synthetic-input-tokens-mean",
                    "--prompt-input-tokens-mean",
                ),
                "isl_stddev": (
                    "--isl-stddev",
                    "--synthetic-input-tokens-stddev",
                    "--prompt-input-tokens-stddev",
                ),
                "isl_block_size": (
                    "--isl-block-size",
                    "--synthetic-input-tokens-block-size",
                    "--prompt-input-tokens-block-size",
                ),
                "prompt_batch_size": ("--batch-size-text", "--prompt-batch-size"),
                "sequence_distribution": ("--seq-dist", "--sequence-distribution"),
                "prefix_prompt_length": (
                    "--prefix-prompt-length",
                    "--prompt-prefix-length",
                ),
                "num_prefix_prompts": (
                    "--prefix-prompt-pool-size",
                    "--prompt-prefix-pool-size",
                    "--num-prefix-prompts",
                ),
            }
            bad = ["/".join(flags) for field, flags in _TEXT.items() if field in s]
            if bad:
                raise ValueError(
                    f"{', '.join(bad)} cannot be used with "
                    f"--endpoint-type {endpoint_type}."
                )
            _TOK = {
                "tokenizer_name": ("--tokenizer",),
                "tokenizer_trust_remote_code": ("--tokenizer-trust-remote-code",),
            }
            bad = ["/".join(flags) for field, flags in _TOK.items() if field in s]
            if bad:
                raise ValueError(
                    f"Tokenizer options ({', '.join(bad)}) cannot be used with "
                    f"--endpoint-type {endpoint_type}."
                )

    # Composed dataset?
    needs_augment = bool(
        {
            "num_prefix_prompts",
            "prefix_prompt_length",
            "shared_system_prompt_length",
            "user_context_prompt_length",
            "image_batch_size",
            "audio_batch_size",
            "video_batch_size",
            "osl_mean",
        }
        & s
    )
    if cli.input_file and needs_augment:
        return _build_composed_dataset(cli, s)

    # Build flat dataset fields
    d: dict[str, Any] = {}
    if "input_file" in s:
        d["path"] = cli.input_file
    if "public_dataset" in s:
        d["name"] = cli.public_dataset
    if "hf_subset" in s:
        d["hf_subset"] = cli.hf_subset
    if "custom_dataset_type" in s:
        d["format"] = cli.custom_dataset_type
    if "dataset_sampling_strategy" in s:
        d["sampling"] = cli.dataset_sampling_strategy
    if "num_dataset_entries" in s:
        d["entries"] = cli.num_dataset_entries

    # Prompts (ISL/OSL)
    prompts: dict[str, Any] = {}
    isl: dict[str, Any] = {}
    if "isl_mean" in s:
        isl["mean"] = cli.isl_mean
    if "isl_stddev" in s:
        isl["stddev"] = cli.isl_stddev
    if isl:
        prompts["isl"] = isl
    osl: dict[str, Any] = {}
    if "osl_mean" in s:
        osl["mean"] = cli.osl_mean
    if "osl_stddev" in s:
        osl["stddev"] = cli.osl_stddev
    if osl:
        prompts["osl"] = osl
    if "isl_block_size" in s:
        prompts["block_size"] = cli.isl_block_size
    if "prompt_batch_size" in s:
        prompts["batch_size"] = cli.prompt_batch_size
    if prompts:
        d["prompts"] = prompts

    # Prefix prompts
    prefix: dict[str, Any] = {}
    if "num_prefix_prompts" in s:
        prefix["pool_size"] = cli.num_prefix_prompts
    if "prefix_prompt_length" in s:
        prefix["length"] = cli.prefix_prompt_length
    if "shared_system_prompt_length" in s:
        prefix["shared_system_length"] = cli.shared_system_prompt_length
    if "user_context_prompt_length" in s:
        prefix["user_context_length"] = cli.user_context_prompt_length
    if prefix:
        d["prefix_prompts"] = prefix

    # Rankings
    rankings: dict[str, Any] = {}
    if "passages_mean" in s:
        rankings.setdefault("passages", {})["mean"] = cli.passages_mean
    if "passages_stddev" in s:
        rankings.setdefault("passages", {})["stddev"] = cli.passages_stddev
    if "passages_prompt_token_mean" in s:
        rankings.setdefault("passage_tokens", {})["mean"] = (
            cli.passages_prompt_token_mean
        )
    if "passages_prompt_token_stddev" in s:
        rankings.setdefault("passage_tokens", {})["stddev"] = (
            cli.passages_prompt_token_stddev
        )
    if "query_prompt_token_mean" in s:
        rankings.setdefault("query_tokens", {})["mean"] = cli.query_prompt_token_mean
    if "query_prompt_token_stddev" in s:
        rankings.setdefault("query_tokens", {})["stddev"] = (
            cli.query_prompt_token_stddev
        )
    if rankings:
        d["rankings"] = rankings

    # Synthesis
    synthesis: dict[str, Any] = {}
    if "synthesis_speedup_ratio" in s:
        synthesis["speedup_ratio"] = cli.synthesis_speedup_ratio
    if "synthesis_prefix_len_multiplier" in s:
        synthesis["prefix_len_multiplier"] = cli.synthesis_prefix_len_multiplier
    if "synthesis_prefix_root_multiplier" in s:
        synthesis["prefix_root_multiplier"] = cli.synthesis_prefix_root_multiplier
    if "synthesis_prompt_len_multiplier" in s:
        synthesis["prompt_len_multiplier"] = cli.synthesis_prompt_len_multiplier
    if "synthesis_max_isl" in s:
        synthesis["max_isl"] = cli.synthesis_max_isl
    if "synthesis_max_osl" in s:
        synthesis["max_osl"] = cli.synthesis_max_osl
    if synthesis:
        d["synthesis"] = synthesis

    # Audio
    audio: dict[str, Any] = {}
    if "audio_length_mean" in s:
        audio.setdefault("length", {})["mean"] = cli.audio_length_mean
    if "audio_length_stddev" in s:
        audio.setdefault("length", {})["stddev"] = cli.audio_length_stddev
    if "audio_batch_size" in s:
        audio["batch_size"] = cli.audio_batch_size
    if "audio_format" in s:
        audio["format"] = cli.audio_format
    if "audio_depths" in s:
        audio["depths"] = cli.audio_depths
    if "audio_sample_rates" in s:
        audio["sample_rates"] = cli.audio_sample_rates
    if "audio_num_channels" in s:
        audio["channels"] = cli.audio_num_channels
    if audio:
        d["audio"] = audio

    # Images
    images: dict[str, Any] = {}
    if "image_height_mean" in s:
        images.setdefault("height", {})["mean"] = cli.image_height_mean
    if "image_height_stddev" in s:
        images.setdefault("height", {})["stddev"] = cli.image_height_stddev
    if "image_width_mean" in s:
        images.setdefault("width", {})["mean"] = cli.image_width_mean
    if "image_width_stddev" in s:
        images.setdefault("width", {})["stddev"] = cli.image_width_stddev
    if "image_batch_size" in s:
        images["batch_size"] = cli.image_batch_size
    if "image_format" in s:
        images["format"] = cli.image_format
    if images:
        d["images"] = images

    # Video
    video: dict[str, Any] = {}
    if "video_batch_size" in s:
        video["batch_size"] = cli.video_batch_size
    if "video_duration" in s:
        video["duration"] = cli.video_duration
    if "video_fps" in s:
        video["fps"] = cli.video_fps
    if "video_width" in s:
        video["width"] = cli.video_width
    if "video_height" in s:
        video["height"] = cli.video_height
    if "video_synth_type" in s:
        video["synth_type"] = cli.video_synth_type
    if "video_format" in s:
        video["format"] = cli.video_format
    if "video_codec" in s:
        video["codec"] = cli.video_codec
    video_audio: dict[str, Any] = {}
    if "video_audio_sample_rate" in s:
        video_audio["sample_rate"] = cli.video_audio_sample_rate
    if "video_audio_num_channels" in s:
        video_audio["channels"] = cli.video_audio_num_channels
    if video_audio:
        video["audio"] = video_audio
    if video:
        d["video"] = video

    # Dataset type
    entries = cli.request_count or cli.num_sessions
    if cli.public_dataset:
        d["type"] = DatasetType.PUBLIC
        if entries is not None:
            d["entries"] = entries
    elif cli.input_file:
        d["type"] = DatasetType.FILE
    else:
        d["type"] = DatasetType.SYNTHETIC
        d.setdefault("entries", entries or cli.num_dataset_entries)
        if needs_text:
            d.setdefault("prompts", {}).setdefault("isl", {}).setdefault("mean", 550)

    # Sequence distribution
    if cli.sequence_distribution:
        from aiperf.common.models.sequence_distribution import DistributionParser

        dist = DistributionParser.parse(cli.sequence_distribution)
        d.setdefault("prompts", {})["sequence_distribution"] = [
            {
                "isl": {
                    "mean": p.input_seq_len,
                    "stddev": p.input_seq_len_stddev,
                },
                "osl": {
                    "mean": p.output_seq_len,
                    "stddev": p.output_seq_len_stddev,
                },
                "probability": p.probability,
            }
            for p in dist.pairs
        ]

    # Turns / delays
    if "num_turns_mean" in s:
        d["turns"] = {"mean": cli.num_turns_mean, "stddev": cli.num_turns_stddev}
    if "turn_delay_mean" in s:
        d["turn_delay"] = {
            "mean": cli.turn_delay_mean,
            "stddev": cli.turn_delay_stddev,
        }
    if "turn_delay_ratio" in s:
        d["turn_delay_ratio"] = cli.turn_delay_ratio

    # Implicit media batch_size
    for media_key, triggers in {
        "images": ("image_width_mean", "image_height_mean", "image_batch_size"),
        "audio": ("audio_length_mean", "audio_batch_size"),
        "video": (
            "video_batch_size",
            "video_width",
            "video_height",
            "video_duration",
            "video_fps",
            "video_synth_type",
        ),
    }.items():
        if (
            media_key in d
            and "batch_size" not in d[media_key]
            and any(f in s for f in triggers)
        ):
            d[media_key]["batch_size"] = 1

    return d


def _build_composed_dataset(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    """File dataset with augmentation overlay."""
    from aiperf.common.enums import DatasetType

    source: dict[str, Any] = {"type": DatasetType.FILE}
    if "input_file" in s:
        source["path"] = cli.input_file
    if "custom_dataset_type" in s:
        source["format"] = cli.custom_dataset_type
    if "dataset_sampling_strategy" in s:
        source["sampling"] = cli.dataset_sampling_strategy

    augment: dict[str, Any] = {}
    if "osl_mean" in s:
        osl: dict[str, Any] = {"mean": cli.osl_mean}
        if "osl_stddev" in s:
            osl["stddev"] = cli.osl_stddev
        augment["osl"] = osl

    if "prefix_prompt_length" in s or "num_prefix_prompts" in s:
        augment["prefix"] = {
            "length": cli.prefix_prompt_length or 128,
            "pool_size": cli.num_prefix_prompts or 1,
        }
    elif "shared_system_prompt_length" in s:
        augment["prefix"] = {
            "length": cli.shared_system_prompt_length,
            "pool_size": 1,
        }
    elif "user_context_prompt_length" in s:
        augment["prefix"] = {"user_context_length": cli.user_context_prompt_length}

    # Media
    images: dict[str, Any] = {}
    if "image_batch_size" in s:
        images["batch_size"] = cli.image_batch_size
    if "image_height_mean" in s:
        images.setdefault("height", {})["mean"] = cli.image_height_mean
    if "image_width_mean" in s:
        images.setdefault("width", {})["mean"] = cli.image_width_mean
    if "image_format" in s:
        images["format"] = cli.image_format
    if images:
        augment["images"] = images

    audio: dict[str, Any] = {}
    if "audio_batch_size" in s:
        audio["batch_size"] = cli.audio_batch_size
    if "audio_length_mean" in s:
        audio.setdefault("length", {})["mean"] = cli.audio_length_mean
    if "audio_format" in s:
        audio["format"] = cli.audio_format
    if audio:
        augment["audio"] = audio

    video: dict[str, Any] = {}
    if "video_batch_size" in s:
        video["batch_size"] = cli.video_batch_size
    if "video_duration" in s:
        video["duration"] = cli.video_duration
    if "video_fps" in s:
        video["fps"] = cli.video_fps
    if video:
        augment["video"] = video

    return {
        "type": DatasetType.COMPOSED,
        "source": source,
        "augment": augment,
        "entries": cli.request_count or cli.num_sessions,
        "random_seed": cli.random_seed,
    }
