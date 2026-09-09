# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-metrics CLI override handling for config-file resolution."""

from __future__ import annotations

import copy
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


def normalize_server_metrics_base_for_override(
    base: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize YAML server_metrics shorthand before CLI override merging.

    The YAML key may be authored under either the canonical ``server_metrics``
    spelling or its documented ``serverMetrics`` camelCase alias -- gating on
    the snake_case key alone leaves the camelCase spelling un-normalized, so
    shorthand fields like ``url`` never get expanded to ``urls`` before
    ``deep_merge`` adds its own ``urls`` key, leaving both to trip
    ``extra="forbid"``.
    """
    if not _has_benchmark_server_metrics_override(overrides):
        return base

    benchmark = base.get("benchmark")
    key = _server_metrics_key(benchmark) if isinstance(benchmark, dict) else None
    if key is None:
        return base

    from aiperf.config.server_metrics import ServerMetricsConfig

    normalized = copy.deepcopy(base)
    normalized_benchmark = normalized["benchmark"]
    normalized_benchmark[key] = ServerMetricsConfig.model_validate(
        normalized_benchmark[key]
    ).model_dump(mode="python")
    return normalized


def _server_metrics_key(benchmark: dict[str, Any]) -> str | None:
    """Return whichever spelling of the server_metrics key is present, if any."""
    from pydantic.alias_generators import to_camel

    for key in ("server_metrics", to_camel("server_metrics")):
        if key in benchmark:
            return key
    return None


def _has_benchmark_server_metrics_override(overrides: dict[str, Any] | None) -> bool:
    benchmark = overrides.get("benchmark") if isinstance(overrides, dict) else None
    return isinstance(benchmark, dict) and "server_metrics" in benchmark
