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
from aiperf.common.models import MetricResult, TelemetryResults
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.display_units_utils import (
    GPU_TELEMETRY_METRICS_CONFIG,
    STAT_KEYS,
    convert_all_metrics_to_display_units,
    normalize_endpoint_display,
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

    TELEMETRY_METRICS_CONFIG = [
        ("GPU Power Usage", "gpu_power_usage", "W"),
        ("Energy Consumption", "energy_consumption", "MJ"),
        ("GPU Utilization", "gpu_utilization", "%"),
        ("GPU Memory Used", "gpu_memory_used", "GB"),
        ("GPU Temperature", "gpu_temperature", "°C"),
        ("SM Clock Frequency", "sm_clock_frequency", "MHz"),
        ("Memory Clock Frequency", "memory_clock_frequency", "MHz"),
    ]

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.debug(lambda: f"Initializing CsvExporter with config: {exporter_config}")
        self._results = exporter_config.results
        self._telemetry_results = exporter_config.telemetry_results
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
        """Export inference and telemetry data to CSV file.

        Creates a CSV file with three sections:
        1. Request metrics (with percentiles)
        2. System metrics (single values)
        3. GPU telemetry metrics (if available)

        Raises:
            Exception: If file writing fails
        """
        self._output_directory.mkdir(parents=True, exist_ok=True)

        self.debug(lambda: f"Exporting data to CSV file: {self._file_path}")

        try:
            records: Mapping[str, MetricResult] = {}
            if self._results.records:
                records = convert_all_metrics_to_display_units(
                    self._results.records, self._metric_registry
                )

            csv_content = self._generate_csv_content(records, self._telemetry_results)

            async with aiofiles.open(
                self._file_path, "w", newline="", encoding="utf-8"
            ) as f:
                await f.write(csv_content)

        except Exception as e:
            self.error(f"Failed to export CSV to {self._file_path}: {e}")
            raise

    def _generate_csv_content(
        self,
        records: Mapping[str, MetricResult],
        telemetry_results: TelemetryResults | None = None,
    ) -> str:
        """Generate CSV content string from inference and telemetry data.

        Args:
            records: Mapping of metric tags to MetricResult objects (inference metrics)
            telemetry_results: Optional GPU telemetry data to include

        Returns:
            str: Complete CSV content with all sections formatted and ready to write
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        request_metrics, system_metrics = self._split_metrics(records)

        if request_metrics:
            self._write_request_metrics(writer, request_metrics)
            if system_metrics:  # blank line between sections
                writer.writerow([])

        if system_metrics:
            self._write_system_metrics(writer, system_metrics)

        # Add telemetry data section if available
        if telemetry_results:
            self._write_telemetry_section(writer, telemetry_results)

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
        records: Mapping[str, MetricResult],
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
        records: Mapping[str, MetricResult],
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

    def _write_telemetry_section(self, writer, telemetry_results) -> None:
        """Write GPU telemetry data section to CSV with statistical aggregation.

        Creates a section for each DCGM endpoint with GPU telemetry metrics.
        For each metric (power, utilization, etc.), writes a subsection with
        statistical summaries (avg, min, max, percentiles) for each GPU.

        Args:
            writer: CSV writer object
            telemetry_results: TelemetryResults containing GPU telemetry data hierarchy
        """

        writer.writerow([])
        writer.writerow([])

        for (
            dcgm_url,
            gpus_data,
        ) in telemetry_results.telemetry_data.dcgm_endpoints.items():
            if not gpus_data:
                continue

            endpoint_display = normalize_endpoint_display(dcgm_url)
            writer.writerow([f"=== GPU Telemetry: {endpoint_display} ==="])
            writer.writerow([])

            for metric_display, metric_key, unit in GPU_TELEMETRY_METRICS_CONFIG:
                has_metric_data = any(
                    self._gpu_has_metric(gpu_data, metric_key)
                    for gpu_data in gpus_data.values()
                )

                if not has_metric_data:
                    continue

                writer.writerow([f"=== {metric_display} ({unit}) ==="])
                header_row = ["GPU Index", "GPU Name", "GPU UUID"]
                header_row.extend([stat.upper() for stat in STAT_KEYS])
                writer.writerow(header_row)

                for gpu_uuid, gpu_data in gpus_data.items():
                    self._write_gpu_metric_row(
                        writer, gpu_data, gpu_uuid, metric_key, metric_display, unit
                    )

                writer.writerow([])

    def _write_gpu_metric_row(
        self, writer, gpu_data, gpu_uuid, metric_key, metric_display, unit
    ):
        """Write a single GPU metric row to CSV with statistical summaries.

        Retrieves metric statistics from gpu_data and writes a row with GPU info
        and statistical values for all stats in STAT_KEYS.

        Args:
            writer: CSV writer object
            gpu_data: GpuTelemetryData containing metric time series
            gpu_uuid: UUID identifier for the GPU
            metric_key: Internal metric name (e.g., "gpu_power_usage")
            metric_display: Display name for the metric (e.g., "GPU Power Usage")
            unit: Unit of measurement (e.g., "W", "GB", "%")
        """
        try:
            metric_result = gpu_data.get_metric_result(
                metric_key, metric_key, metric_display, unit
            )

            row = [
                str(gpu_data.metadata.gpu_index),
                gpu_data.metadata.model_name,
                gpu_uuid,
            ]
            for stat in STAT_KEYS:
                value = getattr(metric_result, stat, None)
                row.append(self._format_number(value))

            writer.writerow(row)
        except Exception as e:
            self.warning(
                f"Failed to write metric row for GPU {gpu_uuid}, metric {metric_key}: {e}"
            )

    def _gpu_has_metric(self, gpu_data, metric_key: str) -> bool:
        """Check if GPU has data for the specified metric.

        Attempts to retrieve metric result to determine if the metric has any data.
        Used to filter out metrics with no collected data.

        Args:
            gpu_data: GpuTelemetryData containing metric time series
            metric_key: Internal metric name to check (e.g., "gpu_power_usage")

        Returns:
            bool: True if metric has data, False if metric is unavailable or has no data
        """
        try:
            gpu_data.get_metric_result(metric_key, metric_key, "test", "test")
            return True
        except Exception as e:
            self.debug(f"GPU metric {metric_key} not available: {e}")
            return False
