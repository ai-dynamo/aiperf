# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any

import aiofiles
from pydantic import BaseModel

from aiperf.common.config import UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import DataExporterType, MetricFlags
from aiperf.common.factories import DataExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import ErrorDetailsCount, MetricResult
from aiperf.common.protocols import DataExporterProtocol
from aiperf.common.types import MetricTagT
from aiperf.exporters.display_units_utils import (
    STAT_KEYS,
    convert_all_metrics_to_display_units,
    normalize_endpoint_display,
)
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_METRICS_CONFIG
from aiperf.metrics.metric_registry import MetricRegistry


class TelemetrySummary(BaseModel):
    """Summary information for telemetry collection."""

    endpoints_tested: list[str]
    endpoints_successful: list[str]
    start_time: datetime
    end_time: datetime


class GpuSummary(BaseModel):
    """Summary of GPU telemetry data."""

    gpu_index: int
    gpu_name: str
    gpu_uuid: str
    hostname: str | None
    metrics: dict[str, dict[str, Any]]  # metric_key -> {stat_key -> value}


class EndpointData(BaseModel):
    """Data for a single endpoint."""

    gpus: dict[str, GpuSummary]


class TelemetryExportData(BaseModel):
    """Telemetry data structure for JSON export."""

    summary: TelemetrySummary
    endpoints: dict[str, EndpointData]


class JsonExportData(BaseModel):
    """Data to be exported to a JSON file."""

    records: dict[MetricTagT, MetricResult] | None = None
    input_config: UserConfig | None = None
    was_cancelled: bool | None = None
    error_summary: list[ErrorDetailsCount] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    telemetry_data: TelemetryExportData | None = None


@DataExporterFactory.register(DataExporterType.JSON)
@implements_protocol(DataExporterProtocol)
class JsonExporter(AIPerfLoggerMixin):
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.debug(lambda: f"Initializing JsonExporter with config: {exporter_config}")
        self._results = exporter_config.results
        self._telemetry_results = exporter_config.telemetry_results
        self._output_directory = exporter_config.user_config.output.artifact_directory
        self._input_config = exporter_config.user_config
        self._metric_registry = MetricRegistry
        self._file_path = (
            self._output_directory / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="JSON Export",
            file_path=self._file_path,
        )

    def _should_export(self, metric: MetricResult) -> bool:
        """Check if a metric should be exported."""
        metric_class = MetricRegistry.get_class(metric.tag)
        res = metric_class.missing_flags(
            MetricFlags.EXPERIMENTAL | MetricFlags.INTERNAL
        )
        self.debug(lambda: f"Metric '{metric.tag}' should be exported: {res}")
        return res

    async def export(self) -> None:
        """Export inference and telemetry data to JSON file.

        Creates a JSON file containing:
        - Input configuration
        - Inference metric results (converted to display units)
        - Telemetry data with statistical summaries per endpoint/GPU
        - Error summaries
        - Timestamps

        Raises:
            Exception: If file writing fails
        """
        self._output_directory.mkdir(parents=True, exist_ok=True)

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

        converted_records: dict[MetricTagT, MetricResult] = {}
        if self._results.records:
            converted_records = convert_all_metrics_to_display_units(
                self._results.records, self._metric_registry
            )
            converted_records = {
                k: v for k, v in converted_records.items() if self._should_export(v)
            }

        telemetry_export_data = None
        if self._telemetry_results:
            summary = TelemetrySummary(
                endpoints_tested=self._telemetry_results.endpoints_tested,
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
            input_config=self._input_config,
            records=converted_records,
            was_cancelled=self._results.was_cancelled,
            error_summary=self._results.error_summary,
            start_time=start_time,
            end_time=end_time,
            telemetry_data=telemetry_export_data,
        )

        self.debug(lambda: f"Exporting data to JSON file: {export_data}")
        export_data_json = export_data.model_dump_json(indent=2, exclude_unset=True)
        async with aiofiles.open(self._file_path, "w") as f:
            await f.write(export_data_json)

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
                ) in GPU_TELEMETRY_METRICS_CONFIG:
                    try:
                        unit = unit_enum.value
                        metric_result = gpu_data.get_metric_result(
                            metric_key, metric_key, metric_key, unit
                        )
                        stats_dict = {}
                        for stat in STAT_KEYS:
                            value = getattr(metric_result, stat, None)
                            stats_dict[stat] = value
                        stats_dict["count"] = metric_result.count
                        stats_dict["unit"] = unit
                        metrics_dict[metric_key] = stats_dict
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
