# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIConfig converter runtime helpers.

Hosts the `build_artifacts` and `build_logging_runtime` helpers on the
`CLIConfig`.

`build_logging_runtime` additionally folds in the five model-level validators
that CLIConfig itself does not carry:

- verbose=True  -> log level DEBUG (and ui=simple in TTY)
- extra_verbose -> log level TRACE (and ui=simple in TTY)
- ui_type defaulting via TTY detection when unset
- zmq_* discriminator -> communication.{type, host/path}
- api_host without api_port raises ValueError
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


def _propagate_set_fields(
    artifacts: dict[str, Any], cli: Any, mapping: dict[str, str]
) -> None:
    """Copy each cli.<src> into artifacts[<dst>] when src is in model_fields_set."""
    cli_set = cli.model_fields_set
    for src, dst in mapping.items():
        if src in cli_set:
            artifacts[dst] = getattr(cli, src)


def build_artifacts(cli: CLIConfig) -> dict[str, Any]:
    """Build the artifacts dict for AIPerfConfig from a CLIConfig.

    Reads the flattened output fields directly off ``cli``. Only emits fields
    the user explicitly set — AIPerfConfig's ArtifactsConfig (Pydantic) supplies
    defaults for any field omitted here, so a stray ``trace=False`` from CLIConfig
    doesn't override a downstream layered default.
    """
    from aiperf.common.enums import ExportFormat, ExportLevel

    artifacts: dict[str, Any] = {}

    _propagate_set_fields(
        artifacts,
        cli,
        {
            "artifact_directory": "dir",
            "export_http_trace": "trace",
            "show_trace_timing": "show_trace_timing",
        },
    )
    cli_set = cli.model_fields_set
    if "slice_duration" in cli_set and cli.slice_duration is not None:
        artifacts["slice_duration"] = cli.slice_duration
    # Only JSONL is wired up for per-record export today (no records-CSV
    # exporter exists). RECORDS/RAW enable it; SUMMARY disables it.
    if cli.export_level in (ExportLevel.RECORDS, ExportLevel.RAW):
        artifacts["records"] = [ExportFormat.JSONL]
    elif "export_level" in cli_set and cli.export_level == ExportLevel.SUMMARY:
        artifacts["records"] = False
    # Only emit raw when the user explicitly set the level OR the level is
    # actually RAW (the CLIConfig default is RECORDS, so an unset field
    # shouldn't noise up the artifacts dict with raw=False).
    if "export_level" in cli_set or cli.export_level == ExportLevel.RAW:
        artifacts["raw"] = cli.export_level == ExportLevel.RAW
    if "profile_export_prefix" in cli_set and cli.profile_export_prefix:
        # If the user passes an absolute path, drop the directory portion so
        # per-run artifact isolation (orchestrator sets artifacts.dir per run)
        # is not bypassed. Suffix stripping happens inside ArtifactsConfig.
        p = cli.profile_export_prefix
        artifacts["prefix"] = p.name if p.is_absolute() else str(p)

    return artifacts


def _apply_runtime_basics(runtime_dict: dict[str, Any], cli: CLIConfig) -> None:
    cli_set = cli.model_fields_set
    if "ui_type" in cli_set:
        runtime_dict["ui"] = cli.ui_type
    if "workers_max" in cli_set and cli.workers_max is not None:
        runtime_dict["workers"] = cli.workers_max
    if (
        "record_processor_service_count" in cli_set
        and cli.record_processor_service_count is not None
    ):
        runtime_dict["record_processors"] = cli.record_processor_service_count
    if "api_port" in cli_set:
        runtime_dict["api_port"] = cli.api_port
    if "api_host" in cli_set:
        runtime_dict["api_host"] = cli.api_host


def _apply_verbosity_and_ui(
    logging_dict: dict[str, Any],
    runtime_dict: dict[str, Any],
    cli: CLIConfig,
) -> None:
    from aiperf.common.enums import AIPerfLogLevel
    from aiperf.common.utils import is_tty
    from aiperf.plugin.enums import UIType

    ui_set = "ui" in runtime_dict
    if cli.extra_verbose:
        logging_dict["level"] = AIPerfLogLevel.TRACE
        runtime_dict["ui"] = UIType.SIMPLE
    elif cli.verbose:
        logging_dict["level"] = AIPerfLogLevel.DEBUG
        runtime_dict["ui"] = UIType.SIMPLE
    elif not ui_set and not is_tty():
        runtime_dict["ui"] = UIType.NONE

    # Dashboard requires a TTY: Textual issues console-setup syscalls that
    # block forever on Windows when stdout is a pipe (e.g. subprocess.PIPE
    # from a test harness, ``aiperf ... | tee``, CI capture). Downgrade to
    # SIMPLE rather than hanging. Applies to every platform — Linux/macOS
    # don't hang, but a non-TTY dashboard renders nothing useful there either.
    if runtime_dict.get("ui") == UIType.DASHBOARD and not is_tty():
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "--ui dashboard requires an interactive terminal; "
            "stdout is not a TTY, falling back to --ui simple"
        )
        runtime_dict["ui"] = UIType.SIMPLE


def _build_communication(cli: CLIConfig) -> dict[str, Any] | None:
    from aiperf.common.enums import CommunicationType

    cli_set = cli.model_fields_set
    if "zmq_ipc_path" in cli_set:
        comm: dict[str, Any] = {"type": CommunicationType.IPC}
        if cli.zmq_ipc_path is not None:
            comm["path"] = str(cli.zmq_ipc_path)
        return comm
    if "zmq_tcp_host" in cli_set:
        return {"type": CommunicationType.TCP, "host": cli.zmq_tcp_host}
    if cli.zmq_dual_bind:
        return {"type": CommunicationType.DUAL}
    return None


def build_logging_runtime(
    cli: CLIConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build (logging, runtime) dicts for AIPerfConfig from a CLIConfig.

    Folds in the five service-runtime validators that CLIConfig strips: verbose/
    extra_verbose log-level promotion, TTY-based ui defaulting, zmq_* ->
    communication discriminator, and the api_host-requires-api_port check.

    Only emits fields the user explicitly set on ``cli`` (per
    ``model_fields_set``); fields the user didn't pass fall through to the
    Pydantic defaults on ``RuntimeConfig`` / ``LoggingConfig``. Verbose-driven
    log-level/UI promotion still writes (it's a derived effect, not a default).
    """
    # api_host requires api_port to be set explicitly (or via env).
    if cli.api_host is not None and cli.api_port is None:
        raise ValueError(
            "api_host requires api_port (or AIPERF_API_SERVER_PORT) to be set"
        )

    logging_dict: dict[str, Any] = {}
    if "log_level" in cli.model_fields_set:
        logging_dict["level"] = cli.log_level
    runtime_dict: dict[str, Any] = {}

    _apply_runtime_basics(runtime_dict, cli)
    _apply_verbosity_and_ui(logging_dict, runtime_dict, cli)
    if (comm := _build_communication(cli)) is not None:
        runtime_dict["communication"] = comm

    return logging_dict, runtime_dict
