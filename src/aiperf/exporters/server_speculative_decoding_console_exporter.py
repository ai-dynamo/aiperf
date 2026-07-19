# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from rich.markup import escape
from rich.table import Table

from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.finite import is_finite_value
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.server_metrics_models import (
    GaugeMetricData,
    GaugeSeries,
)
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.utils import normalize_endpoint_display

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


class SGLangSpeculativeDecodingMetric(NamedTuple):
    """SGLang speculative decoding gauge display configuration."""

    name: str
    display_name: str
    scale: float
    precision: int


class SpeculativeDecodingSeries(NamedTuple):
    """SGLang speculative decoding gauge series with source endpoint."""

    endpoint: str
    series: GaugeSeries


class ServerSpeculativeDecodingConsoleExporter(AIPerfLoggerMixin):
    """Console exporter for SGLang speculative decoding server metrics."""

    _COLUMNS: tuple[str, ...] = ("mean", "min", "max", "p50", "p90")
    _SPEC_METRICS: tuple[SGLangSpeculativeDecodingMetric, ...] = (
        SGLangSpeculativeDecodingMetric(
            name="sglang:spec_accept_rate",
            display_name="Accept Rate (%)",
            scale=100.0,
            precision=1,
        ),
        SGLangSpeculativeDecodingMetric(
            name="sglang:spec_accept_length",
            display_name="Accept Length",
            scale=1.0,
            precision=2,
        ),
    )

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._server_metrics_results = exporter_config.server_metrics_results
        self._model_names = {
            model_name.lower() for model_name in exporter_config.cfg.get_model_names()
        }
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
        for metric in self._SPEC_METRICS:
            series_list = self._collect_gauge_series(metric.name)
            if not series_list:
                continue
            for index, source in enumerate(series_list, start=1):
                if source.series.stats is None:
                    continue
                row = self._build_row(metric, source, index, series_list)
                if row is None:
                    self.warning(
                        lambda metric=metric, source=source: (
                            "Skipping SGLang speculative decoding console row "
                            f"for {metric.name}: non-finite gauge summary values "
                            f"encountered for labels {source.series.labels or {}}"
                        )
                    )
                    continue
                rows.append(row)
        return rows

    def _collect_gauge_series(
        self, metric_name: str
    ) -> list[SpeculativeDecodingSeries]:
        if self._server_metrics_results is None:
            return []

        summaries = self._server_metrics_results.endpoint_summaries or {}
        series_list: list[SpeculativeDecodingSeries] = []
        for endpoint_summary in summaries.values():
            endpoint = normalize_endpoint_display(endpoint_summary.endpoint_url)
            metric_data = endpoint_summary.metrics.get(metric_name)
            if not isinstance(metric_data, GaugeMetricData):
                continue
            for series in metric_data.series:
                if series.stats is not None and self._matches_model(series.labels):
                    series_list.append(
                        SpeculativeDecodingSeries(
                            endpoint=endpoint,
                            series=series,
                        )
                    )
        return series_list

    def _matches_model(self, labels: dict[str, str] | None) -> bool:
        if labels is None:
            return False
        model_name = labels.get("model_name")
        if model_name is None or model_name.lower() not in self._model_names:
            return False
        # SGLang exposes these gauges with rank labels, but the values describe
        # scheduler acceptance state rather than per-rank work. With all
        # scheduler metrics enabled, non-zero PP/TP ranks duplicate the same
        # signal, so the console view keeps only the leader series.
        return labels.get("pp_rank", "0") == "0" and labels.get("tp_rank", "0") == "0"

    def _build_row(
        self,
        metric: SGLangSpeculativeDecodingMetric,
        source: SpeculativeDecodingSeries,
        index: int,
        series_list: list[SpeculativeDecodingSeries],
    ) -> SpeculativeDecodingRow | None:
        series = source.series
        if series.stats is None:
            return None
        mean = self._scaled_stat(series.stats.avg, metric.scale)
        min_value = self._scaled_stat(series.stats.min, metric.scale)
        max_value = self._scaled_stat(series.stats.max, metric.scale)
        p50 = self._scaled_stat(series.stats.p50, metric.scale)
        p90 = self._scaled_stat(series.stats.p90, metric.scale)
        if (
            mean is None
            or min_value is None
            or max_value is None
            or p50 is None
            or p90 is None
        ):
            return None
        return SpeculativeDecodingRow(
            metric=self._row_label(metric.display_name, source, index, series_list),
            mean=mean,
            min_value=min_value,
            max_value=max_value,
            p50=p50,
            p90=p90,
            precision=metric.precision,
        )

    @staticmethod
    def _scaled_stat(value: float | None, scale: float) -> float | None:
        if not is_finite_value(value):
            return None
        scaled = float(value) * scale
        if not is_finite_value(scaled):
            return None
        return scaled

    def _row_label(
        self,
        display_name: str,
        source: SpeculativeDecodingSeries,
        index: int,
        series_list: list[SpeculativeDecodingSeries],
    ) -> str:
        if len(series_list) == 1:
            return display_name
        labels = source.series.labels or {}
        suffix_parts: list[str] = []
        if len({item.endpoint for item in series_list}) > 1:
            suffix_parts.append(self._label_part("endpoint", source.endpoint))
        if self._has_multiple_label_values(series_list, "model_name"):
            suffix_parts.append(self._label_part("model_name", labels["model_name"]))
        suffix_parts.extend(
            self._label_part(key, value)
            for key, value in sorted(labels.items())
            if key not in {"model_name", "pp_rank", "tp_rank"}
        )
        suffix = ", ".join(suffix_parts) if suffix_parts else f"series={index}"
        return f"{display_name} ({suffix})"

    @staticmethod
    def _label_part(key: str, value: str) -> str:
        return f"{escape(key)}={escape(value)}"

    @staticmethod
    def _has_multiple_label_values(
        series_list: list[SpeculativeDecodingSeries], label_name: str
    ) -> bool:
        values = {
            item.series.labels[label_name]
            for item in series_list
            if item.series.labels is not None and label_name in item.series.labels
        }
        return len(values) > 1

    @staticmethod
    def _format(value: float, precision: int) -> str:
        return f"{value:,.{precision}f}"
