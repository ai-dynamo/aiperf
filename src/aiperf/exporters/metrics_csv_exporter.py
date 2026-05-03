# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import io
import numbers
from collections.abc import Mapping, Sequence
from decimal import Decimal

from aiperf.common.constants import STAT_KEYS
from aiperf.common.models import GpuSummary, MetricResult
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.gpu_telemetry.constants import get_gpu_telemetry_metrics_config


def _percentile_keys_from(stat_keys: Sequence[str]) -> list[str]:
    # e.g., ["avg","min","max","p50","p90","p95","p99"] -> ["p50","p90","p95","p99"]
    return [k for k in stat_keys if len(k) >= 2 and k[0] == "p" and k[1:].isdigit()]


class MetricsCsvExporter(MetricsBaseExporter):
    """Exports records to a CSV file in a legacy, two-section format."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self._file_path = exporter_config.user_config.output.profile_export_csv_file
        self._percentile_keys = _percentile_keys_from(STAT_KEYS)
        self.trace_or_debug(
            lambda: f"Initializing MetricsCsvExporter with config: {exporter_config}",
            lambda: f"Initializing MetricsCsvExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="CSV Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate CSV content string from inference and telemetry data.

        Uses instance data members self._results.records and self._telemetry_results.

        Returns:
            str: Complete CSV content with all sections formatted and ready to write
        """
        buf = io.StringIO()
        writer = csv.writer(buf)

        # Use base class method to prepare metrics
        prepared_metrics = self._prepare_metrics(self._results.records)

        request_metrics, system_metrics = self._split_metrics(prepared_metrics)

        if request_metrics:
            self._write_request_metrics(writer, request_metrics)
            if system_metrics:  # blank line between sections
                writer.writerow([])

        if system_metrics:
            self._write_system_metrics(writer, system_metrics)

        # Add telemetry data section if available
        if self._telemetry_results:
            self._write_telemetry_section(writer)

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
            row = [self._format_metric_name(metric)]
            for stat_name in STAT_KEYS:
                value = getattr(metric, stat_name, None)
                row.append(self._format_number(value))
            writer.writerow(row)

    def _write_system_metrics(
        self,
        writer: csv.writer,
        records: Mapping[str, MetricResult],
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

    def _get_optional_headers_and_fields(
        self, *header_names: str
    ) -> tuple[list[str], list[str]]:
        """Get optional headers and fields from GPU summaries. Returns a tuple of (optional_headers, optional_fields).

        Args:
            header_names: List of header names to get optional headers and fields for.
        Returns:
            A tuple of (optional_headers, optional_fields).

        Example:
            For a GPU summary with hostname "gpu-0", namespace "default", and pod name "pod-0":
            ```python
            >>> optional_headers, optional_fields = self._get_optional_headers_and_fields("Hostname", "Namespace", "Pod Name")
            >>> print(optional_headers)
            ["Hostname", "Namespace", "Pod Name"]
            >>> print(optional_fields)
            ["hostname", "namespace", "pod_name"]
            ```

            For a GPU summary with only hostname "gpu-0":
            ```python
            >>> optional_headers, optional_fields = self._get_optional_headers_and_fields("Hostname", "Namespace", "Pod Name")
            >>> print(optional_headers)
            ["Hostname"]
            >>> print(optional_fields)
            ["hostname"]
            ```
        """
        headers_found: dict[str, bool] = {header: False for header in header_names}
        field_names: dict[str, str] = {
            header: header.lower().replace(" ", "_") for header in header_names
        }
        for endpoint_data in self._telemetry_results.endpoints.values():
            for gpu_summary in endpoint_data.gpus.values():
                for header, field in field_names.items():
                    if getattr(gpu_summary, field, None) is not None:
                        headers_found[header] = True

        optional_headers = [header for header, found in headers_found.items() if found]
        optional_fields = [
            field_names[header] for header, found in headers_found.items() if found
        ]
        return optional_headers, optional_fields

    def _write_telemetry_section(self, writer: csv.writer) -> None:
        """Write GPU telemetry data section to CSV in structured table format.

        Uses self._telemetry_results (TelemetryExportData) instance data member.

        Creates a single flat table with all GPU telemetry metrics that's easy to
        parse programmatically for visualization platforms (pandas, Tableau, Excel, etc.).

        Each row represents one metric for one GPU with all statistics in columns.

        Args:
            writer: CSV writer object
        """

        writer.writerow([])
        writer.writerow([])

        # Write header row for GPU telemetry table
        header_row = [
            "Endpoint",
            "GPU_Index",
            "GPU_Name",
            "GPU_UUID",
        ]
        optional_headers, optional_fields = self._get_optional_headers_and_fields(
            "Hostname", "Namespace", "Pod Name"
        )
        header_row.extend(["Metric", *STAT_KEYS])
        header_row.extend(optional_headers)
        writer.writerow(header_row)

        # TelemetryExportData uses: endpoints[endpoint_display] -> EndpointData.gpus[gpu_key] -> GpuSummary
        for (
            endpoint_display,
            endpoint_data,
        ) in self._telemetry_results.endpoints.items():
            if not endpoint_data.gpus:
                continue

            for _, gpu_summary in endpoint_data.gpus.items():
                for (
                    metric_display,
                    metric_key,
                    unit_enum,
                ) in get_gpu_telemetry_metrics_config():
                    # Check if metric exists in pre-computed metrics
                    if metric_key not in gpu_summary.metrics:
                        continue

                    self._write_gpu_metric_row_from_summary(
                        writer,
                        endpoint_display,
                        gpu_summary,
                        optional_fields,
                        metric_key,
                        metric_display,
                        unit_enum.value,
                    )

    def _write_gpu_metric_row_from_summary(
        self,
        writer: csv.writer,
        endpoint_display: str,
        gpu_summary: GpuSummary,
        optional_fields: list[str],
        metric_key: str,
        metric_display: str,
        unit: str,
    ) -> None:
        """Write a single GPU metric row from pre-computed GpuSummary.

        Each row contains: endpoint, GPU info, metric name with unit, and all stats.
        This format is optimized for programmatic extraction and visualization.

        Args:
            writer: CSV writer object
            endpoint_display: Display name of the endpoint
            gpu_summary: GpuSummary with pre-computed metrics (from TelemetryExportData)
            optional_fields: List of optional fields to write to the row
            metric_key: Internal metric name (e.g., "gpu_power_usage")
            metric_display: Display name for the metric (e.g., "GPU Power Usage")
            unit: Unit of measurement (e.g., "W", "GB", "%")
        """
        try:
            metric_result = gpu_summary.metrics[metric_key]

            # Format metric name with unit like inference metrics
            metric_with_unit = f"{metric_display} ({unit})"

            row = [
                endpoint_display,
                str(gpu_summary.gpu_index),
                gpu_summary.gpu_name,
                gpu_summary.gpu_uuid,
                metric_with_unit,
            ]

            for stat in STAT_KEYS:
                value = getattr(metric_result, stat, None)
                row.append(self._format_number(value))

            for field in optional_fields:
                value = getattr(gpu_summary, field, None)
                row.append(str(value))

            writer.writerow(row)
        except Exception as e:
            self.warning(
                f"Failed to write metric row for GPU {gpu_summary.gpu_uuid}, metric {metric_key}: {e}"
            )
