# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricConsoleGroup
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter


class ConsoleNvidiaPowerEfficiencyExporter(ConsoleMetricsExporter):
    """Console exporter for NVIDIA cross-GPU power efficiency totals.

    Renders the `nvidia_*` efficiency totals (power, energy, tokens/joule,
    energy/user) in their own vendor-attributed table instead of the main
    metrics table. The totals are computed from `nvidia_power_usage` /
    `nvidia_energy_consumption`, populated identically by the DCGM and pynvml
    collectors. When no NVIDIA GPU reported, the metrics are absent, so the base
    `get_renderable` returns `None` and the section is omitted entirely.
    """

    title = "GPU Power Efficiency (NVIDIA)"
    console_groups = (MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA,)
    split_by_group = False
    # These totals are single aggregate values, not distributions, so only the
    # average column is meaningful; the percentile/min/max/std columns would all
    # be N/A.
    STAT_COLUMN_KEYS = ["avg"]


class ConsoleAmdPowerEfficiencyExporter(ConsoleMetricsExporter):
    """Console exporter for AMD cross-GPU power efficiency totals.

    AMD-side counterpart to `ConsoleNvidiaPowerEfficiencyExporter`. Renders the
    `amd_*` efficiency totals computed from `amd_power` / `amd_energy_consumption`
    (amdsmi collector). Omitted entirely when no AMD GPU reported.
    """

    title = "GPU Power Efficiency (AMD)"
    console_groups = (MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD,)
    split_by_group = False
    STAT_COLUMN_KEYS = ["avg"]
