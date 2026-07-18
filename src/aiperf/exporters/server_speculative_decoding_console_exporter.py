# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from statistics import fmean
from typing import TYPE_CHECKING, Any, NamedTuple

from rich.table import Table

from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.server_metrics_models import GaugeMetricData, GaugeStats
from aiperf.exporters.exporter_config import ExporterConfig

if TYPE_CHECKING:
    from rich.console import Console


class SpeculativeDecodingRow(NamedTuple):
    """Single speculative decoding row for console output."""

    metric: str
    mean: float
    min_value: float
    max_value: float
    p50: float
    p90: float
    precision: int


class ServerSpeculativeDecodingConsoleExporter(AIPerfLoggerMixin):
    """Console exporter for SGLang speculative decoding server metrics."""

    _COLUMNS: tuple[str, ...] = ("mean", "min", "max", "p50", "p90")
    _SPEC_METRICS: tuple[tuple[str, str, float, int], ...] = (
        ("sglang:spec_accept_rate", "SGLang Spec Accept Rate (%)", 100.0, 1),
        ("sglang:spec_accept_length", "SGLang Spec Accept Length", 1.0, 2),
    )

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._server_metrics_results = exporter_config.server_metrics_results
        self._rows = self._build_rows()
        if not self._rows:
            raise ConsoleExporterDisabled(
                "Speculative decoding console exporter is disabled: "
                "no SGLang speculative decoding metrics"
            )

    async def export(self, console: Console) -> None:
        """Render SGLang speculative decoding server metrics."""
        table = Table(
            title="NVIDIA AIPerf | Server Metrics: Speculative Decoding",
            show_header=True,
        )
        table.add_column("Metric", style="cyan")
        for column in self._COLUMNS:
            table.add_column(column, justify="right", style="green")

        for row in self._rows:
            table.add_row(
                row.metric,
                self._format(row.mean, row.precision),
                self._format(row.min_value, row.precision),
                self._format(row.max_value, row.precision),
                self._format(row.p50, row.precision),
                self._format(row.p90, row.precision),
            )

        console.print()
        console.print(table)

    def _build_rows(self) -> list[SpeculativeDecodingRow]:
        if (
            self._server_metrics_results is None
            or not self._server_metrics_results.endpoint_summaries
        ):
            return []

        rows: list[SpeculativeDecodingRow] = []
        for metric_name, display_name, scale, precision in self._SPEC_METRICS:
            stats = self._collect_gauge_stats(metric_name)
            if not stats:
                continue
            values = self._summarize_stats(stats, scale)
            if values is None:
                continue
            rows.append(
                SpeculativeDecodingRow(
                    metric=display_name,
                    mean=values[0],
                    min_value=values[1],
                    max_value=values[2],
                    p50=values[3],
                    p90=values[4],
                    precision=precision,
                )
            )
        return rows

    def _collect_gauge_stats(self, metric_name: str) -> list[GaugeStats]:
        if self._server_metrics_results is None:
            return []

        summaries = self._server_metrics_results.endpoint_summaries or {}
        stats: list[GaugeStats] = []
        for endpoint_summary in summaries.values():
            metric_data = endpoint_summary.metrics.get(metric_name)
            if not isinstance(metric_data, GaugeMetricData):
                continue
            for series in metric_data.series:
                if series.stats is not None:
                    stats.append(series.stats)
        return stats

    def _summarize_stats(
        self, stats: list[GaugeStats], scale: float
    ) -> tuple[float, float, float, float, float] | None:
        mean_values = self._values(stats, "avg", scale)
        min_values = self._values(stats, "min", scale)
        max_values = self._values(stats, "max", scale)
        p50_values = self._values(stats, "p50", scale)
        p90_values = self._values(stats, "p90", scale)
        if not (
            mean_values and min_values and max_values and p50_values and p90_values
        ):
            return None
        return (
            fmean(mean_values),
            min(min_values),
            max(max_values),
            fmean(p50_values),
            fmean(p90_values),
        )

    @staticmethod
    def _values(stats: list[GaugeStats], field: str, scale: float) -> list[float]:
        return [
            value * scale
            for stat in stats
            if (value := getattr(stat, field)) is not None
        ]

    @staticmethod
    def _format(value: float, precision: int) -> str:
        return f"{value:,.{precision}f}"
