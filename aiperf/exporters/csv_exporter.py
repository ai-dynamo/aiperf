# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
from collections.abc import Iterable, Mapping

import aiofiles

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType
from aiperf.common.factories import DataExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.display_units_utils import (
    STAT_KEYS,
    convert_all_metrics_to_display_units,
)
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.metric_registry import MetricRegistry


def _percentile_keys_from(stat_keys: Iterable[str]) -> list[str]:
    # e.g., ["avg","min","max","p50","p90","p95","p99"] -> ["p50","p90","p95","p99"]
    return [k for k in stat_keys if len(k) >= 2 and k[0] == "p" and k[1:].isdigit()]


@DataExporterFactory.register(DataExporterType.CSV)
@implements_protocol(DataExporterProtocol)
class CsvExporter(AIPerfLoggerMixin):
    """Exports records to a CSV file in a legacy, two-section format."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.debug(lambda: f"Initializing CsvExporter with config: {exporter_config}")
        self._results = exporter_config.results
        self._output_directory = exporter_config.user_config.output.artifact_directory
        self._metric_registry = MetricRegistry()
        self._percentile_keys = _percentile_keys_from(STAT_KEYS)

    async def export(self) -> None:
        filename = self._output_directory / "profile_export_aiperf.csv"
        self._output_directory.mkdir(parents=True, exist_ok=True)

        self.debug(lambda: f"Exporting data to CSV file: {filename}")

        try:
            records: Mapping[str, MetricResult] = {}
            if self._results.records:
                records = convert_all_metrics_to_display_units(
                    self._results.records, self._metric_registry
                )

            csv_content = self._generate_csv_content(records)

            async with aiofiles.open(filename, "w", newline="", encoding="utf-8") as f:
                await f.write(csv_content)

        except Exception as e:
            self.error(f"Failed to export CSV to {filename}: {e}")
            raise

    def _generate_csv_content(self, records: Mapping[str, MetricResult]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)

        request_metrics, system_metrics = self._split_metrics(records)

        if request_metrics:
            self._write_request_metrics(writer, request_metrics)
            if system_metrics:  # blank line between sections
                writer.writerow([])

        if system_metrics:
            self._write_system_metrics(writer, system_metrics)

        return buf.getvalue()

    def _split_metrics(
        self, records: Mapping[str, MetricResult]
    ) -> tuple[dict[str, MetricResult], dict[str, MetricResult]]:
        """Split metrics into request metrics (with percentiles) and system metrics (single values)."""
        request_metrics: dict[str, MetricResult] = {}
        system_metrics: dict[str, MetricResult] = {}

        for tag, metric in records.items():
            if self._has_percentiles(metric):
                request_metrics[tag] = metric
            else:
                system_metrics[tag] = metric

        return request_metrics, system_metrics

    def _has_percentiles(self, metric: MetricResult) -> bool:
        """Check if a metric has any percentile data."""
        return any(getattr(metric, k, None) is not None for k in self._percentile_keys)

    def _write_request_metrics(
        self, writer: csv.writer, records: Mapping[str, MetricResult]
    ) -> None:
        header = ["Metric"] + list(STAT_KEYS)
        writer.writerow(header)

        for _, metric in sorted(records.items(), key=lambda kv: kv[0]):
            row = [self._format_metric_name(metric)]
            for stat_name in STAT_KEYS:
                value = getattr(metric, stat_name, None)
                row.append(self._format_number(value))
            writer.writerow(row)

    def _write_system_metrics(
        self, writer: csv.writer, records: Mapping[str, MetricResult]
    ) -> None:
        writer.writerow(["Metric", "Value"])
        for _, metric in sorted(records.items(), key=lambda kv: kv[0]):
            writer.writerow(
                [self._format_metric_name(metric), self._format_number(metric.avg)]
            )

    def _format_metric_name(self, metric: MetricResult) -> str:
        """Format metric name with its unit."""
        name = metric.header or ""
        if metric.unit and metric.unit.lower() not in {"count", "requests"}:
            name = f"{name} ({metric.unit})" if name else f"({metric.unit})"
        return name

    def _format_number(self, value) -> str:
        """Format a number for CSV output."""
        if value is None:
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{float(value):.2f}"
        return str(value)
