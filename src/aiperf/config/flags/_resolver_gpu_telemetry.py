# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-telemetry CLI override handling for config-file resolution."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.config.flags import CLIConfig


def build_gpu_telemetry_override(cli: CLIConfig) -> dict[str, Any] | None:
    """Build only explicit GPU-telemetry CLI overrides for config-file mode."""
    fields_set = cli.model_fields_set & {"gpu_telemetry", "no_gpu_telemetry"}
    if not fields_set:
        return None

    from aiperf.config.flags._converter_telemetry import build_gpu_telemetry

    built = build_gpu_telemetry(cli)
    override: dict[str, Any] = {}
    if "no_gpu_telemetry" in fields_set:
        override["enabled"] = built["enabled"]
    elif "gpu_telemetry" in fields_set:
        override.update(built)

    return override or None


def normalize_gpu_telemetry_base_for_override(
    base: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize YAML GPU telemetry shorthand before CLI override merging."""
    if not _has_benchmark_gpu_telemetry_override(overrides):
        return base

    benchmark = base.get("benchmark")
    if not isinstance(benchmark, dict) or "gpu_telemetry" not in benchmark:
        return base

    from aiperf.config.gpu_telemetry import GpuTelemetryConfig

    normalized = copy.deepcopy(base)
    normalized_benchmark = normalized["benchmark"]
    normalized_benchmark["gpu_telemetry"] = GpuTelemetryConfig.model_validate(
        normalized_benchmark["gpu_telemetry"]
    ).model_dump(mode="python")
    return normalized


def _has_benchmark_gpu_telemetry_override(overrides: dict[str, Any] | None) -> bool:
    benchmark = overrides.get("benchmark") if isinstance(overrides, dict) else None
    return isinstance(benchmark, dict) and "gpu_telemetry" in benchmark
