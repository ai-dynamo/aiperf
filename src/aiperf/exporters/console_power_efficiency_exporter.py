# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import MetricConsoleGroup
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter


class ConsolePowerEfficiencyExporter(ConsoleMetricsExporter):
    """Console exporter for NVIDIA cross-GPU power efficiency totals.

    Renders `total_gpu_power`, `total_gpu_energy`, `output_tokens_per_joule`, and
    `energy_per_user` in their own vendor-attributed table instead of the main
    metrics table. The totals are NVIDIA-sourced today (computed from
    `nvidia_power_usage` / `nvidia_energy_consumption`, populated identically by
    both the DCGM and pynvml collectors). When GPU telemetry is disabled the
    metrics are absent, so the base `get_renderable` returns `None` and the
    section is omitted entirely.
    """

    title = "GPU Power Efficiency (NVIDIA)"
    console_groups = (MetricConsoleGroup.GPU_POWER_EFFICIENCY,)
    split_by_group = False
