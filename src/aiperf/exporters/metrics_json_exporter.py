# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from datetime import datetime

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType
from aiperf.common.factories import DataExporterFactory
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import (
    EndpointData,
    GpuSummary,
    JsonExportData,
    JsonMetricResult,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.common.protocols import DataExporterProtocol
from aiperf.exporters.display_units_utils import normalize_endpoint_display
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.gpu_telemetry.constants import get_gpu_telemetry_metrics_config


@DataExporterFactory.register(DataExporterType.JSON)
@implements_protocol(DataExporterProtocol)
class MetricsJsonExporter(MetricsBaseExporter):
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self._file_path = exporter_config.user_config.output.profile_export_json_file
        self.trace_or_debug(
            lambda: f"Initializing MetricsJsonExporter with config: {exporter_config}",
            lambda: f"Initializing MetricsJsonExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="JSON Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate JSON content string from inference and telemetry data.

        Uses instance data members self._results.records and self._telemetry_results.

        Returns:
            str: Complete JSON content with all sections formatted and ready to write
        """
        # Use helper method to prepare metrics
        prepared_json_metrics = self._prepare_metrics_for_json(self._results.records)

        start_time = (
            datetime.fromtimestamp(self._results.start_ns / NANOS_PER_SECOND)
            if self._results.start_ns
            else None
        )
        end_time = (
            datetime.fromtimestamp(self._results.end_ns / NANOS_PER_SECOND)
            if self._results.end_ns
            else None
        )

        telemetry_export_data = None
        if self._telemetry_results:
            summary = TelemetrySummary(
                endpoints_configured=self._telemetry_results.endpoints_configured,
                endpoints_successful=self._telemetry_results.endpoints_successful,
                start_time=datetime.fromtimestamp(
                    self._telemetry_results.start_ns / NANOS_PER_SECOND
                ),
                end_time=datetime.fromtimestamp(
                    self._telemetry_results.end_ns / NANOS_PER_SECOND
                ),
            )
            telemetry_export_data = TelemetryExportData(
                summary=summary,
                endpoints=self._generate_telemetry_statistical_summary(),
            )

        export_data = JsonExportData(
            input_config=self._user_config,
            was_cancelled=self._results.was_cancelled,
            error_summary=self._results.error_summary,
            start_time=start_time,
            end_time=end_time,
            telemetry_data=telemetry_export_data,
        )

        # Add all prepared metrics dynamically
        for metric_tag, json_result in prepared_json_metrics.items():
            setattr(export_data, metric_tag, json_result)

        self.trace_or_debug(
            lambda: f"Exporting data to JSON file: {export_data}",
            lambda: f"Exporting data to JSON file: {self._file_path}",
        )
        return export_data.model_dump_json(
            indent=2, exclude_unset=True, exclude_none=True
        )

    def _prepare_metrics_for_json(
        self, metric_results: Iterable[MetricResult]
    ) -> dict[str, JsonMetricResult]:
        """Prepare and convert metrics to JsonMetricResult objects.

        Applies unit conversion, filtering, and conversion to JSON format.

        Args:
            metric_results: Raw metric results to prepare

        Returns:
            dict mapping metric tags to JsonMetricResult objects ready for export
        """
        prepared = self._prepare_metrics(metric_results)
        return {tag: result.to_json_result() for tag, result in prepared.items()}

    def _generate_telemetry_statistical_summary(self) -> dict[str, EndpointData]:
        """Generate clean statistical summary of telemetry data for JSON export.

        Processes telemetry hierarchy into a structured dict with:
        - Endpoints organized by normalized display name (e.g., "localhost:9400")
        - GPU data with metadata (index, name, UUID, hostname)
        - Metric statistics (avg, min, max, p99, p90, p75, std, count) per GPU
        - Only includes metrics with available data

        Returns:
            dict: Nested structure of endpoints -> gpus -> metrics with statistics.
                Empty dict if no telemetry data available.
        """
        summary = {}

        if not self._telemetry_results or not self._telemetry_results.telemetry_data:
            return summary

        for (
            dcgm_url,
            gpus_data,
        ) in self._telemetry_results.telemetry_data.dcgm_endpoints.items():
            endpoint_display = normalize_endpoint_display(dcgm_url)
            gpus_dict = {}

            for gpu_uuid, gpu_data in gpus_data.items():
                metrics_dict = {}

                for (
                    _metric_display,
                    metric_key,
                    unit_enum,
                ) in get_gpu_telemetry_metrics_config():
                    try:
                        unit = unit_enum.value
                        metric_result = gpu_data.get_metric_result(
                            metric_key, metric_key, metric_key, unit
                        )
                        metrics_dict[metric_key] = metric_result.to_json_result()
                    except Exception:
                        continue

                gpu_summary = GpuSummary(
                    gpu_index=gpu_data.metadata.gpu_index,
                    gpu_name=gpu_data.metadata.model_name,
                    gpu_uuid=gpu_uuid,
                    hostname=gpu_data.metadata.hostname,
                    metrics=metrics_dict,
                )

                gpus_dict[f"gpu_{gpu_data.metadata.gpu_index}"] = gpu_summary

            summary[endpoint_display] = EndpointData(gpus=gpus_dict)

        return summary
