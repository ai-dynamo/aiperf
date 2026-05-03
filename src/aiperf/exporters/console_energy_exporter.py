# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Console exporter for energy efficiency metrics."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table

from aiperf.analysis.energy_analyzer import EnergyEfficiencySummary
from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig


class ConsoleEnergyExporter(AIPerfLoggerMixin):
    """Console exporter that renders energy efficiency metrics as a Rich table."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        if exporter_config.energy_efficiency_results is None:
            raise ConsoleExporterDisabled("No energy efficiency results available")
        self._summary: EnergyEfficiencySummary = (
            exporter_config.energy_efficiency_results
        )

    async def export(self, console: Console) -> None:
        if not self._summary.metric_results:
            return

        s = self._summary

        table = Table(
            title="NVIDIA AIPerf | Energy Efficiency Metrics", box=box.SIMPLE_HEAVY
        )
        table.add_column("Metric", justify="right", style="cyan")
        table.add_column("Value", justify="right", style="green")
        table.add_column("Unit", justify="left")

        _row(table, "Total GPU Energy", s.total_gpu_energy_j, "J")
        _row(table, "Average GPU Power", s.average_gpu_power_w, "W")
        _row(table, "Energy Per Output Token", s.energy_per_output_token_mj, "mJ/token")
        _row(table, "Energy Per Total Token", s.energy_per_total_token_mj, "mJ/token")
        _row(table, "Energy Per Request", s.energy_per_request_j, "J/req")
        _row(table, "Output TPS Per Watt", s.output_tps_per_watt, "tps/W")
        _row(table, "Performance Per Watt", s.performance_per_watt, "req/s/W")
        _row(table, "Goodput Per Watt", s.goodput_per_watt, "good-req/s/W")

        info = Table.grid(padding=(0, 1))
        info.add_row(
            f"[bold]GPU Count:[/bold] {s.gpu_count}",
            f"[bold]Energy Source:[/bold] {s.energy_source.value}",
        )

        console.print("\n")
        console.print(info)
        console.print(table)
        console.file.flush()


def _row(table: Table, label: str, value: float | None, unit: str) -> None:
    """Add a row, skipping None values."""
    if value is None:
        return
    table.add_row(label, f"{value:,.2f}", unit)
