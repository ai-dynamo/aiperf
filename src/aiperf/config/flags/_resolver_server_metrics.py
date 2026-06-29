# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-metrics CLI override handling for config-file resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


def build_server_metrics_override(cli: CLIConfig) -> dict[str, Any] | None:
    """Build only explicit server-metrics CLI overrides for config-file mode.

    The CLI-only ``build_server_metrics`` builder intentionally emits a complete
    section, including default formats and an empty URL list. In config-file
    mode, only user-set fields should overlay YAML so ``--server-metrics-formats``
    can replace YAML formats without clobbering YAML URLs.
    """
    fields_set = cli.model_fields_set & {
        "server_metrics",
        "server_metrics_formats",
        "no_server_metrics",
    }
    if not fields_set:
        return None

    from aiperf.config.flags._converter_telemetry import build_server_metrics

    built = build_server_metrics(cli)
    override: dict[str, Any] = {}
    if "no_server_metrics" in fields_set:
        override["enabled"] = built["enabled"]
    elif "server_metrics" in fields_set:
        override["enabled"] = True
        override["urls"] = built["urls"]

    if "server_metrics_formats" in fields_set and "formats" in built:
        override["enabled"] = True
        override["formats"] = built["formats"]

    return override or None
