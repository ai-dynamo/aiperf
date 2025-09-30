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
        """
        Initialize the CsvExporter with configuration and prepare internal export state.
        
        Sets exporter results, optional telemetry results, output directory, metric registry,
        default CSV file path, and percentile keys used when formatting exported CSV content.
        
        Parameters:
            exporter_config (ExporterConfig): Exporter configuration containing `results` (metrics to export),
                `user_config.output.artifact_directory` (destination directory), and optionally
                `telemetry_results` (GPU telemetry data). Additional keyword arguments are passed to the superclass.
        """
        super().__init__(**kwargs)
        self.debug(lambda: f"Initializing CsvExporter with config: {exporter_config}")
        self._results = exporter_config.results
        self._telemetry_results = getattr(exporter_config, "telemetry_results", None)
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
        """
        Write collected metrics and optional telemetry results to the configured CSV file.
        
        This method ensures the output directory exists, converts stored metric records to display units when present, generates CSV content (including an optional GPU telemetry section), and writes the content to the exporter file path. Any exception raised during generation or file I/O is logged and re-raised.
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
        self, records: Mapping[str, MetricResult], telemetry_results=None
    ) -> str:
        """
        Generate the CSV file content for the provided metric records and optional telemetry results.
        
        Parameters:
            records (Mapping[str, MetricResult]): Mapping of metric tag to MetricResult to include in the CSV.
            telemetry_results (optional): Telemetry results object; when provided, a GPU telemetry section is appended to the CSV.
        
        Returns:
            csv_content (str): Complete CSV-formatted content including request metrics, system metrics, and an optional telemetry section.
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
        """
        Partition metric records into request-style metrics that include percentiles and system-style metrics that do not.
        
        Parameters:
            records (Mapping[str, MetricResult]): Mapping of metric tag to MetricResult objects to classify.
        
        Returns:
            tuple[dict[str, MetricResult], dict[str, MetricResult]]: A tuple (request_metrics, system_metrics) where `request_metrics` maps tags to metrics that have percentile values and `system_metrics` maps tags to metrics that do not.
        """
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
        """
        Write the request-style metrics section to the CSV writer using STAT_KEYS as the column headers.
        
        Writes a header row of "Metric" followed by STAT_KEYS, then emits one row per metric (sorted by metric tag) that passes the exporter filter. Each row starts with the formatted metric name and contains the metric's STAT_KEYS values formatted for display.
        
        Parameters:
        	writer (csv.writer): CSV writer to receive header and metric rows.
        	records (Mapping[str, MetricResult]): Mapping of metric tag to MetricResult for request metrics.
        """
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
        """
        Write the system-level metrics section to the CSV output.
        
        Writes a header row ["Metric", "Value"] and then, for each metric in `records` sorted by tag, writes a row containing the formatted metric name and its formatted average value. Metrics that are not eligible for export are skipped.
        
        Parameters:
            writer (csv.writer): CSV writer used to emit rows.
            records (Mapping[str, MetricResult]): Mapping of metric tag to MetricResult objects to export.
        """
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
        """
        Convert a value into a CSV-friendly string representation.
        
        Parameters:
            value: The value to format; may be None, bool, Integral, Real, Decimal, or any other type.
        
        Returns:
            A string suitable for CSV output: an empty string for `None`, `"True"`/`"False"` for booleans, an integer string for integral values, a floating representation with two decimal places for real numbers and `Decimal`, and `str(value)` for all other types.
        """
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
        """
        Write GPU telemetry sections to the CSV for each telemetry endpoint that contains data.
        
        For each endpoint in telemetry_results.telemetry_data.dcgm_endpoints, emits a labeled section that lists selected GPU metrics (power, energy, utilization, memory, temperature, SM and memory clocks) when data for that metric exists across any GPU. Each metric section contains a header row and rows per GPU with the following columns: "GPU Index", "GPU Name", "GPU UUID", "Avg", "Min", "Max", "P99", "P90", "P75", "Std". Endpoints with no GPUs or metrics with no data are skipped. Individual GPU entries that fail to produce a metric result are omitted.
        """

        writer.writerow([])
        writer.writerow([])

        for (
            dcgm_url,
            gpus_data,
        ) in telemetry_results.telemetry_data.dcgm_endpoints.items():
            if not gpus_data:
                continue

            endpoint_display = dcgm_url.replace("http://", "").replace("/metrics", "")
            writer.writerow([f"=== GPU Telemetry: {endpoint_display} ==="])
            writer.writerow([])

            metrics_to_export = [
                ("GPU Power Usage", "gpu_power_usage", "W"),
                ("Energy Consumption", "energy_consumption", "MJ"),
                ("GPU Utilization", "gpu_utilization", "%"),
                ("GPU Memory Used", "gpu_memory_used", "GB"),
                ("GPU Temperature", "gpu_temperature", "°C"),
                ("SM Clock Frequency", "sm_clock_frequency", "MHz"),
                ("Memory Clock Frequency", "memory_clock_frequency", "MHz"),
            ]

            for metric_display, metric_key, unit in metrics_to_export:
                has_metric_data = any(
                    self._gpu_has_metric(gpu_data, metric_key)
                    for gpu_data in gpus_data.values()
                )

                if not has_metric_data:
                    continue

                writer.writerow([f"=== {metric_display} ({unit}) ==="])
                writer.writerow(
                    [
                        "GPU Index",
                        "GPU Name",
                        "GPU UUID",
                        "Avg",
                        "Min",
                        "Max",
                        "P99",
                        "P90",
                        "P75",
                        "Std",
                    ]
                )

                for gpu_uuid, gpu_data in gpus_data.items():
                    try:
                        metric_result = gpu_data.get_metric_result(
                            metric_key, metric_key, metric_display, unit
                        )

                        writer.writerow(
                            [
                                str(gpu_data.metadata.gpu_index),
                                gpu_data.metadata.model_name,
                                gpu_uuid,
                                self._format_number(metric_result.avg),
                                self._format_number(metric_result.min),
                                self._format_number(metric_result.max),
                                self._format_number(metric_result.p99),
                                self._format_number(metric_result.p90),
                                self._format_number(metric_result.p75),
                                self._format_number(metric_result.std),
                            ]
                        )
                    except Exception:
                        continue

                writer.writerow([])

    def _gpu_has_metric(self, gpu_data, metric_key: str) -> bool:
        """
        Determine whether the given GPU data contains a result for the specified metric key.
        
        Parameters:
            gpu_data: An object exposing get_metric_result(metric_key, ...) used to probe for the metric.
            metric_key (str): The metric identifier to check for on the GPU.
        
        Returns:
            bool: `True` if a MetricResult can be retrieved for `metric_key`, `False` otherwise.
        """
        try:
            gpu_data.get_metric_result(metric_key, metric_key, "test", "test")
            return True
        except Exception:
            return False
