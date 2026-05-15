# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry section builders for the ``CLIConfig`` -> ``AIPerfConfig`` converter.

Builds ``gpu_telemetry``, ``server_metrics``, ``otel``, and ``mlflow`` sections
by reading top-level fields on the ``CLIConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


def _url(item: str) -> str:
    return item if item.startswith("http") else f"http://{item}"


def _local_collector_keywords() -> dict[str, Any]:
    """CLI keyword (plugin name, lowercased) -> ``GPUTelemetryCollectorType``.

    Derived from plugin metadata: any ``gpu_telemetry_collector`` plugin whose
    metadata declares ``is_local: true`` becomes a valid keyword. Adding a new
    local collector therefore only requires editing ``plugins.yaml`` — no edits
    here.
    """
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import GPUTelemetryCollectorType

    return {
        member.lower(): GPUTelemetryCollectorType(member)
        for member in GPUTelemetryCollectorType
        if plugins.get_gpu_telemetry_collector_metadata(member).is_local
    }


def build_gpu_telemetry(cli: CLIConfig) -> dict[str, Any]:
    """Translate ``--gpu-telemetry`` magic-list into the telemetry dict.

    Classifies each ``--gpu-telemetry`` item into a collector type, URLs,
    optional metrics file, or dashboard mode. Local collector keywords are
    discovered from plugin metadata (``is_local: true``) so adding a new
    local backend never touches this converter.
    """
    from aiperf.common.enums import GPUTelemetryMode
    from aiperf.plugin.enums import GPUTelemetryCollectorType

    if cli.no_gpu_telemetry:
        return {"enabled": False}
    if not cli.gpu_telemetry:
        return {"enabled": True}

    local_keywords = _local_collector_keywords()
    urls: list[str] = []
    metrics_file: Path | None = None
    collector_type = cli._gpu_telemetry_collector_type
    mode = cli._gpu_telemetry_mode

    for item in cli.gpu_telemetry:
        lowered = item.lower()
        if lowered in local_keywords:
            selected = local_keywords[lowered]
            # Reject mixing two different local collectors in the same call.
            if (
                collector_type != GPUTelemetryCollectorType.DCGM
                and collector_type != selected
            ):
                raise ValueError(
                    "Conflicting local GPU telemetry collectors: "
                    f"'{collector_type}' and '{selected}'. Choose exactly one."
                )
            collector_type = selected
        elif lowered == "dashboard":
            mode = GPUTelemetryMode.REALTIME_DASHBOARD
        elif item.endswith(".csv"):
            metrics_file = Path(item)
        elif item.startswith("http") or ":" in item:
            urls.append(_url(item))
        else:
            valid_kw = ", ".join(f"'{k}'" for k in sorted(local_keywords))
            raise ValueError(
                f"Invalid GPU telemetry item: {item}. Valid options are: "
                f"{valid_kw}, 'dashboard', '.csv' file, and URLs."
            )

    cli._gpu_telemetry_collector_type = collector_type
    cli._gpu_telemetry_mode = mode

    gpu_telemetry: dict[str, Any] = {
        "enabled": True,
        "urls": urls,
        "collector": collector_type,
        "mode": mode,
    }
    if metrics_file is not None:
        gpu_telemetry["metrics_file"] = metrics_file
    return gpu_telemetry


def build_server_metrics(cli: CLIConfig) -> dict[str, Any]:
    """Translate ``--server-metrics`` flags into the server-metrics dict."""
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


def build_otel(cli: CLIConfig) -> dict[str, Any]:
    """Translate OTel CLI flags into the first-class OTel config dict."""
    otel: dict[str, Any] = {}
    cli_set = cli.model_fields_set
    if "otel_url" in cli_set:
        otel["metrics_url"] = cli.otel_url
    if "stream" in cli_set:
        otel["stream_metrics_enabled"] = cli.stream in ("default", "metrics")
        otel["stream_timing_enabled"] = cli.stream in ("default", "timing")
    if "gen_ai_provider" in cli_set:
        otel["gen_ai_provider"] = cli.gen_ai_provider
    return otel


def build_mlflow(cli: CLIConfig) -> dict[str, Any]:
    """Translate MLflow CLI flags into the first-class MLflow config dict."""
    mapping = {
        "mlflow_tracking_uri": "tracking_uri",
        "mlflow_experiment": "experiment",
        "mlflow_run_name": "run_name",
        "mlflow_tags": "tags",
        "mlflow_parent_run_id": "parent_run_id",
        "mlflow_artifact_globs": "artifact_globs",
    }
    return {
        dst: getattr(cli, src)
        for src, dst in mapping.items()
        if src in cli.model_fields_set
    }
