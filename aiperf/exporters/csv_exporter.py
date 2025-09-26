# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import numbers
from collections.abc import Mapping, Sequence
from decimal import Decimal

import aiofiles

from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType
from aiperf.common.enums.metric_enums import MetricFlags
from aiperf.common.factories import DataExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.display_units_utils import (
    STAT_KEYS,
    convert_all_metrics_to_display_units,
)
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.metrics.metric_registry import MetricRegistry


def _percentile_keys_from(stat_keys: Sequence[str]) -> list[str]:
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
        self._metric_registry = MetricRegistry
        self._file_path = (
            self._output_directory / OutputDefaults.PROFILE_EXPORT_AIPERF_CSV_FILE
        )
        self._percentile_keys = _percentile_keys_from(STAT_KEYS)

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="CSV Export",
            file_path=self._file_path,
        )

    async def export(self) -> None:
        self._output_directory.mkdir(parents=True, exist_ok=True)

        self.debug(lambda: f"Exporting data to CSV file: {self._file_path}")

        try:
            records: Mapping[str, MetricResult] = {}
            if self._results.records:
                records = convert_all_metrics_to_display_units(
                    self._results.records, self._metric_registry
                )

            csv_content = self._generate_csv_content(records)

            async with aiofiles.open(
                self._file_path, "w", newline="", encoding="utf-8"
            ) as f:
                await f.write(csv_content)

        except Exception as e:
            self.error(f"Failed to export CSV to {self._file_path}: {e}")
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
        self,
        writer: csv.writer,
        records: Mapping[str, MetricResult],  # type: ignore
    ) -> None:
        header = ["Metric"] + list(STAT_KEYS)
        writer.writerow(header)

        for _, metric in sorted(records.items(), key=lambda kv: kv[0]):
            if not self._should_export(metric):
                continue
            row = [self._format_metric_name(metric)]
            for stat_name in STAT_KEYS:
                value = getattr(metric, stat_name, None)
                row.append(self._format_number(value))
            writer.writerow(row)

    def _should_export(self, metric: MetricResult) -> bool:
        """Check if a metric should be exported."""
        metric_class = MetricRegistry.get_class(metric.tag)
        res = metric_class.missing_flags(
            MetricFlags.EXPERIMENTAL | MetricFlags.INTERNAL
        )
        self.debug(lambda: f"Metric '{metric.tag}' should be exported: {res}")
        return res

    def _write_system_metrics(
        self,
        writer: csv.writer,
        records: Mapping[str, MetricResult],  # type: ignore
    ) -> None:
        writer.writerow(["Metric", "Value"])
        for _, metric in sorted(records.items(), key=lambda kv: kv[0]):
            if not self._should_export(metric):
                continue
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
        # Handle bools explicitly (bool is a subclass of int)
        if isinstance(value, bool):
            return str(value)
        # Integers (covers built-in int and other Integral implementations)
        if isinstance(value, numbers.Integral):
            return f"{int(value)}"
        # Real numbers (covers built-in float and many Real implementations) and Decimal
        if isinstance(value, numbers.Real | Decimal):
            return f"{float(value):.2f}"

        return str(value)
