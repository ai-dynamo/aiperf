# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry section builders for the v1 ``UserConfig`` -> v2 ``AIPerfConfig`` converter.

Ported from ``aiperf.config._cli_sections.build_gpu_telemetry`` /
``build_server_metrics`` with reads rerouted from the legacy CLI dataclass to
the v1 ``UserConfig`` (top-level fields).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.v1 import UserConfig


def _url(item: str) -> str:
    return item if item.startswith("http") else f"http://{item}"


def build_gpu_telemetry(user: UserConfig) -> dict[str, Any]:
    """Translate v1 ``--gpu-telemetry`` magic-list into the v2 telemetry dict."""
    if user.no_gpu_telemetry:
        return {"enabled": False}
    if not user.gpu_telemetry:
        return {"enabled": True}
    urls: list[str] = []
    metrics_file: Path | None = None
    for item in user.gpu_telemetry:
        if item.endswith(".csv"):
            metrics_file = Path(item)
        elif item.startswith("http") or ":" in item:
            urls.append(_url(item))
    gpu_telemetry: dict[str, Any] = {"enabled": True, "urls": urls}
    if metrics_file is not None:
        gpu_telemetry["metrics_file"] = metrics_file
    return gpu_telemetry


def build_server_metrics(user: UserConfig) -> dict[str, Any]:
    """Translate v1 ``--server-metrics`` flags into the v2 server-metrics dict."""
    from aiperf.common.metric_utils import normalize_metrics_endpoint_url

    if user.no_server_metrics:
        return {"enabled": False}
    sm_urls = [
        normalize_metrics_endpoint_url(_url(i))
        for i in user.server_metrics or []
        if i.startswith("http") or ":" in i
    ]
    server_metrics: dict[str, Any] = {"enabled": True, "urls": sm_urls}
    if user.server_metrics_formats:
        server_metrics["formats"] = list(user.server_metrics_formats)
    return server_metrics
