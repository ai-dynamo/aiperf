# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 converter runtime helpers.

Ports the `build_artifacts` and `build_logging_runtime` helpers from the
legacy `_cli_sections.py` onto the v1 `UserConfig` + `ServiceConfig` pair.

`build_logging_runtime` additionally folds in the four model-level validators
that v1 ServiceConfig strips relative to origin/main:

- verbose=True  -> log level DEBUG (and ui=simple in TTY)
- extra_verbose -> log level TRACE (and ui=simple in TTY)
- ui_type defaulting via TTY detection when unset
- zmq_* discriminator -> communication.{type, host/path}
- api_host without api_port raises ValueError
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.v1 import ServiceConfig, UserConfig


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


def build_artifacts(user: UserConfig) -> dict[str, Any]:
    """Build the artifacts dict for AIPerfConfig from a v1 UserConfig.

    Reads `user.output` (OutputConfig) and the synthesized cli_command from
    sys.argv. Only emits fields the user explicitly set on OutputConfig — v2's
    ArtifactsConfig (Pydantic) supplies defaults for any field omitted here, so
    a stray ``trace=False`` from v1 doesn't override a downstream layered
    default. ``cli_command`` is always synthesized (not user input).
    """
    from aiperf.common.enums import ExportFormat, ExportLevel

    artifacts: dict[str, Any] = {"cli_command": _build_cli_command()}

    output = user.output
    if output is None:
        return artifacts

    out_set = output.model_fields_set
    if "artifact_directory" in out_set:
        artifacts["dir"] = output.artifact_directory
    if "slice_duration" in out_set and output.slice_duration is not None:
        artifacts["slice_duration"] = output.slice_duration
    if "export_http_trace" in out_set:
        artifacts["trace"] = output.export_http_trace
    if "export_per_chunk_data" in out_set:
        artifacts["per_chunk_data"] = output.export_per_chunk_data
    if "show_trace_timing" in out_set:
        artifacts["show_trace_timing"] = output.show_trace_timing
    if "export_level" in out_set:
        if output.export_level in (ExportLevel.RECORDS, ExportLevel.RAW):
            artifacts["records"] = [ExportFormat.JSONL, "csv"]
        artifacts["raw"] = output.export_level == ExportLevel.RAW
    if "profile_export_prefix" in out_set and output.profile_export_prefix:
        artifacts["prefix"] = Path(output.profile_export_prefix).stem

    return artifacts


def _apply_runtime_basics(runtime_dict: dict[str, Any], service: ServiceConfig) -> None:
    svc_set = service.model_fields_set
    if "ui_type" in svc_set:
        runtime_dict["ui"] = service.ui_type
    if (
        service.workers is not None
        and "max" in service.workers.model_fields_set
        and service.workers.max is not None
    ):
        runtime_dict["workers"] = service.workers.max
    if (
        "record_processor_service_count" in svc_set
        and service.record_processor_service_count is not None
    ):
        runtime_dict["record_processors"] = service.record_processor_service_count


def _apply_verbosity_and_ui(
    logging_dict: dict[str, Any],
    runtime_dict: dict[str, Any],
    service: ServiceConfig,
) -> None:
    from aiperf.common.enums import AIPerfLogLevel
    from aiperf.common.utils import is_tty
    from aiperf.plugin.enums import UIType

    ui_set = "ui" in runtime_dict
    if service.extra_verbose:
        logging_dict["level"] = AIPerfLogLevel.TRACE
        runtime_dict["ui"] = UIType.SIMPLE
    elif service.verbose:
        logging_dict["level"] = AIPerfLogLevel.DEBUG
        runtime_dict["ui"] = UIType.SIMPLE
    elif not ui_set and not is_tty():
        runtime_dict["ui"] = UIType.NONE


def _build_communication(service: ServiceConfig) -> dict[str, Any] | None:
    from aiperf.common.enums import CommunicationType

    if service.zmq_ipc is not None:
        comm: dict[str, Any] = {"type": CommunicationType.IPC}
        if service.zmq_ipc.path is not None:
            comm["path"] = str(service.zmq_ipc.path)
        return comm
    if service.zmq_tcp is not None:
        return {"type": CommunicationType.TCP, "host": service.zmq_tcp.host}
    if service.zmq_dual_bind is not None:
        return {"type": CommunicationType.DUAL}
    return None


def build_logging_runtime(
    user: UserConfig, service: ServiceConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (logging, runtime) dicts for AIPerfConfig from v1 inputs.

    Folds in the four ServiceConfig validators that v1 strips: verbose/
    extra_verbose log-level promotion, TTY-based ui defaulting, zmq_* ->
    communication discriminator, and the api_host-requires-api_port check.

    Only emits fields the user explicitly set on ``service`` (per
    ``model_fields_set``); fields the user didn't pass fall through to v2's
    Pydantic defaults on ``RuntimeConfig`` / ``LoggingConfig``. Verbose-driven
    log-level/UI promotion still writes (it's a derived effect, not a default).
    """
    # Fold validate_api_host_requires_port from origin/main ServiceConfig.
    if service.api_host is not None and service.api_port is None:
        raise ValueError(
            "api_host requires api_port (or AIPERF_API_SERVER_PORT) to be set"
        )

    logging_dict: dict[str, Any] = {}
    if "log_level" in service.model_fields_set:
        logging_dict["level"] = service.log_level
    runtime_dict: dict[str, Any] = {}

    _apply_runtime_basics(runtime_dict, service)
    _apply_verbosity_and_ui(logging_dict, runtime_dict, service)
    if (comm := _build_communication(service)) is not None:
        runtime_dict["communication"] = comm

    return logging_dict, runtime_dict
