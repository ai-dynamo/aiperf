# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Section-builders for the CLI-to-config converter.

Split from ``cli_converter.py`` so each top-level section stays small enough
for the ergonomics line budget and is independently testable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel


def _url(item: str) -> str:
    return item if item.startswith("http") else f"http://{item}"


def _endpoint_template_from_extra(
    endpoint: dict[str, Any], extra: dict[str, Any]
) -> None:
    payload_template = extra.pop("payload_template", None)
    if payload_template is None:
        return
    path = Path(payload_template)
    body = path.read_text() if path.is_file() else payload_template
    endpoint["template"] = {
        "body": body,
        "response_field": extra.pop("response_field", "text"),
    }


def _endpoint_template_fallback(endpoint: dict[str, Any]) -> None:
    from aiperf.plugin.enums import EndpointType

    if endpoint.get("type") != EndpointType.TEMPLATE or "template" in endpoint:
        return
    extra_raw = endpoint.get("extra")
    if not extra_raw:
        return
    ex = dict(extra_raw) if isinstance(extra_raw, list) else extra_raw
    ts = ex.get("payload_template")
    if ts is None:
        return
    tp = Path(ts)
    endpoint["template"] = {"body": tp.read_text() if tp.is_file() else ts}


def build_endpoint(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    endpoint: dict[str, Any] = {"urls": [_url(u) for u in cli.urls]}
    mapping = {
        "url_selection_strategy": "url_strategy",
        "endpoint_type": "type",
        "streaming": "streaming",
        "custom_endpoint": "path",
        "api_key": "api_key",
        "request_timeout_seconds": "timeout",
        "ready_check_timeout": "ready_check_timeout",
        "transport_type": "transport",
        "use_legacy_max_tokens": "use_legacy_max_tokens",
        "use_server_token_count": "use_server_token_count",
        "connection_reuse_strategy": "connection_reuse",
    }
    for field, key in mapping.items():
        if field in s:
            endpoint[key] = getattr(cli, field)
    if cli.headers:
        endpoint["headers"] = dict(cli.headers)
    if cli.extra_inputs:
        extra = dict(cli.extra_inputs)
        _endpoint_template_from_extra(endpoint, extra)
        endpoint["extra"] = extra
    _endpoint_template_fallback(endpoint)
    return endpoint


def build_models(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    models: dict[str, Any] = {"items": [{"name": n} for n in cli.model_names]}
    if "model_selection_strategy" in s:
        models["strategy"] = cli.model_selection_strategy
    return models


_PROF_FIELD_MAP = {
    "benchmark_duration": "duration",
    "benchmark_grace_period": "grace_period",
    "concurrency": "concurrency",
    "prefill_concurrency": "prefill_concurrency",
    "arrival_smoothness": "smoothness",
    "request_count": "requests",
    "num_sessions": "sessions",
    "num_users": "users",
    "request_rate": "rate",
    "user_centric_rate": "rate",
    "fixed_schedule_auto_offset": "auto_offset",
    "fixed_schedule_start_offset": "start_offset",
    "fixed_schedule_end_offset": "end_offset",
}


def _profiling_phase_type(cli: BaseModel) -> Any:
    from aiperf.config.phases import PhaseType
    from aiperf.plugin.enums import ArrivalPattern

    if cli.fixed_schedule:
        return PhaseType.FIXED_SCHEDULE
    if cli.user_centric_rate is not None:
        return PhaseType.USER_CENTRIC
    if cli.request_rate is not None:
        match cli.arrival_pattern:
            case ArrivalPattern.GAMMA:
                return PhaseType.GAMMA
            case ArrivalPattern.CONSTANT:
                return PhaseType.CONSTANT
            case _:
                return PhaseType.POISSON
    return PhaseType.CONCURRENCY


_RAMP_FIELDS = {
    "concurrency_ramp_duration": "concurrency_ramp",
    "prefill_concurrency_ramp_duration": "prefill_ramp",
    "request_rate_ramp_duration": "rate_ramp",
}


def _apply_profiling_ramps(prof: dict[str, Any], cli: BaseModel, s: set[str]) -> None:
    for field, key in _RAMP_FIELDS.items():
        if field in s:
            prof[key] = {"duration": getattr(cli, field)}


def _validate_profiling(prof: dict[str, Any], cli: BaseModel) -> None:
    from aiperf.config.phases import PhaseType

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


def build_profiling(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    from aiperf.config.phases import PhaseType

    prof: dict[str, Any] = {}
    for field, key in _PROF_FIELD_MAP.items():
        if field in s:
            prof[key] = getattr(cli, field)
    _apply_profiling_ramps(prof, cli, s)

    prof["type"] = _profiling_phase_type(cli)
    if prof["type"] == PhaseType.FIXED_SCHEDULE and "start_offset" in prof:
        prof.setdefault("auto_offset", False)

    _validate_profiling(prof, cli)
    return prof


def _warmup_count_field(w: dict[str, Any], cli: BaseModel) -> None:
    if cli.warmup_request_count is not None:
        w["requests"] = cli.warmup_request_count
    elif cli.warmup_num_sessions is not None:
        w["sessions"] = cli.warmup_num_sessions
    elif cli.warmup_duration is not None:
        w["duration"] = cli.warmup_duration


def _warmup_pattern_type(w: dict[str, Any], cli: BaseModel, s: set[str]) -> None:
    from aiperf.config.phases import PhaseType
    from aiperf.plugin.enums import ArrivalPattern

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


def _warmup_ramps(w: dict[str, Any], cli: BaseModel, s: set[str]) -> None:
    def _pick(warmup_field: str, fallback_field: str) -> Any:
        if warmup_field in s:
            return getattr(cli, warmup_field)
        if fallback_field in s:
            return getattr(cli, fallback_field)
        return None

    cr = _pick("warmup_concurrency_ramp_duration", "concurrency_ramp_duration")
    pr = _pick(
        "warmup_prefill_concurrency_ramp_duration",
        "prefill_concurrency_ramp_duration",
    )
    rr = _pick("warmup_request_rate_ramp_duration", "request_rate_ramp_duration")
    if cr is not None:
        w["concurrency_ramp"] = {"duration": cr}
    if pr is not None:
        w["prefill_ramp"] = {"duration": pr}
    if rr is not None:
        w["rate_ramp"] = {"duration": rr}


def build_warmup(cli: BaseModel, s: set[str]) -> dict[str, Any] | None:
    if not ({"warmup_request_count", "warmup_num_sessions", "warmup_duration"} & s):
        return None
    w: dict[str, Any] = {"exclude_from_results": True}
    _warmup_count_field(w, cli)
    _warmup_pattern_type(w, cli, s)
    _warmup_ramps(w, cli, s)
    if cli.warmup_prefill_concurrency is not None:
        w["prefill_concurrency"] = cli.warmup_prefill_concurrency
    if cli.warmup_grace_period is not None:
        w["grace_period"] = cli.warmup_grace_period
    return w


def _redact_args(args: list[Any]) -> list[Any]:
    from aiperf.common.redact import REDACTED_VALUE

    _sensitive_tokens = ("api-key", "api_key", "authorization", "token")
    out: list[Any] = []
    redact_next = False
    for arg in args:
        if redact_next:
            out.append(REDACTED_VALUE)
            redact_next = False
            continue
        if isinstance(arg, str) and arg.startswith("-"):
            name = arg.lstrip("-").lower()
            key, _, inline = name.partition("=")
            if any(tok in key for tok in _sensitive_tokens):
                if inline:
                    out.append(f"{arg.split('=', 1)[0]}={REDACTED_VALUE}")
                else:
                    out.append(arg)
                    redact_next = True
                continue
        out.append(arg)
    return out


def _build_cli_command() -> str:
    from aiperf.config.parsing import coerce_value

    args = [coerce_value(x) for x in sys.argv[1:]]
    redacted = _redact_args(args)
    return " ".join(
        ["aiperf"]
        + [
            f"'{x}'"
            if isinstance(x, str) and not x.startswith("-") and x != "profile"
            else str(x)
            for x in redacted
        ]
    )


def build_artifacts(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    from aiperf.common.enums import ExportFormat, ExportLevel

    artifacts: dict[str, Any] = {"cli_command": _build_cli_command()}
    simple_map = {
        "artifact_directory": "dir",
        "slice_duration": "slice_duration",
        "export_http_trace": "trace",
        "export_per_chunk_data": "per_chunk_data",
        "show_trace_timing": "show_trace_timing",
    }
    for field, key in simple_map.items():
        if field in s:
            artifacts[key] = getattr(cli, field)
    if cli.export_level in (ExportLevel.RECORDS, ExportLevel.RAW):
        artifacts["records"] = [ExportFormat.JSONL, "csv"]
    artifacts["raw"] = cli.export_level == ExportLevel.RAW
    if cli.profile_export_prefix:
        artifacts["prefix"] = Path(cli.profile_export_prefix).stem
    return artifacts


def build_gpu_telemetry(cli: BaseModel) -> dict[str, Any]:
    if cli.no_gpu_telemetry:
        return {"enabled": False}
    if not cli.gpu_telemetry:
        return {"enabled": True}
    urls: list[str] = []
    metrics_file: Path | None = None
    for item in cli.gpu_telemetry:
        if item.endswith(".csv"):
            metrics_file = Path(item)
        elif item.startswith("http") or ":" in item:
            urls.append(_url(item))
    gpu_telemetry: dict[str, Any] = {"enabled": True, "urls": urls}
    if metrics_file is not None:
        gpu_telemetry["metrics_file"] = metrics_file
    return gpu_telemetry


def build_server_metrics(cli: BaseModel) -> dict[str, Any]:
    from aiperf.common.metric_utils import normalize_metrics_endpoint_url

    if cli.no_server_metrics:
        return {"enabled": False}
    sm_urls = [
        normalize_metrics_endpoint_url(_url(i))
        for i in cli.server_metrics or []
        if i.startswith("http") or ":" in i
    ]
    server_metrics: dict[str, Any] = {"enabled": True, "urls": sm_urls}
    if cli.server_metrics_formats:
        server_metrics["formats"] = list(cli.server_metrics_formats)
    return server_metrics


def build_logging_runtime(
    cli: BaseModel, s: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from aiperf.common.enums import AIPerfLogLevel, CommunicationType
    from aiperf.common.utils import is_tty
    from aiperf.plugin.enums import UIType

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
    return logging_dict, runtime_dict


def build_tokenizer(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    tok: dict[str, Any] = {}
    if "tokenizer_name" in s:
        tok["name"] = cli.tokenizer_name
    if "tokenizer_revision" in s:
        tok["revision"] = cli.tokenizer_revision
    if "tokenizer_trust_remote_code" in s:
        tok["trust_remote_code"] = cli.tokenizer_trust_remote_code
    return tok


def build_accuracy(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    mapping = {
        "accuracy_benchmark": "benchmark",
        "accuracy_tasks": "tasks",
        "accuracy_n_shots": "n_shots",
        "accuracy_enable_cot": "enable_cot",
        "accuracy_grader": "grader",
        "accuracy_system_prompt": "system_prompt",
        "accuracy_verbose": "verbose",
    }
    acc: dict[str, Any] = {}
    for field, key in mapping.items():
        if field in s:
            acc[key] = getattr(cli, field)
    return acc


def build_multi_run(cli: BaseModel, s: set[str]) -> dict[str, Any]:
    mapping = {
        "num_profile_runs": "num_runs",
        "profile_run_cooldown_seconds": "cooldown_seconds",
        "confidence_level": "confidence_level",
        "profile_run_disable_warmup_after_first": "disable_warmup_after_first",
        "set_consistent_seed": "set_consistent_seed",
        "convergence_metric": "convergence_metric",
        "convergence_mode": "convergence_mode",
        "convergence_threshold": "convergence_threshold",
        "convergence_stat": "convergence_stat",
    }
    mr: dict[str, Any] = {}
    for field, key in mapping.items():
        if field in s:
            mr[key] = getattr(cli, field)
    return mr
