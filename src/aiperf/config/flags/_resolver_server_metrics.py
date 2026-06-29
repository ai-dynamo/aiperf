# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-metrics CLI override handling for config-file resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


def apply_server_metrics_overrides(out: dict[str, Any], cli: CLIConfig) -> None:
    """Map explicit server-metrics CLI flags onto the YAML server_metrics block.

    ``build_server_metrics`` always emits the full CLI-only section, including an
    empty ``urls`` list and default ``formats``. For config-file mode we only
    overlay the fields the user actually passed so ``--server-metrics-formats``
    can replace YAML formats without clobbering YAML URLs.
    """
    fields_set = cli.model_fields_set & {
        "server_metrics",
        "server_metrics_formats",
        "no_server_metrics",
    }
    if not fields_set:
        return

    from aiperf.config.flags._converter_telemetry import build_server_metrics

    built = build_server_metrics(cli)
    server_metrics: dict[str, Any] = {}
    if "no_server_metrics" in fields_set:
        server_metrics["enabled"] = built["enabled"]
    elif "server_metrics" in fields_set:
        server_metrics["enabled"] = True
        server_metrics["urls"] = built["urls"]

    if "server_metrics_formats" in fields_set:
        server_metrics["formats"] = built["formats"]

    if server_metrics:
        out["server_metrics"] = server_metrics
