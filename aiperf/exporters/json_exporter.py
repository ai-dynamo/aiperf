# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

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
from aiperf.exporters.display_units_utils import convert_all_metrics_to_display_units
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.metrics.metric_registry import MetricRegistry


class JsonExportData(BaseModel):
    """Data to be exported to a JSON file."""

    records: dict[MetricTagT, MetricResult] | None = None
    input_config: UserConfig | None = None
    was_cancelled: bool | None = None
    error_summary: list[ErrorDetailsCount] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    telemetry_data: dict | None = None  # Raw telemetry data for JSON export


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
        self._telemetry_results = getattr(exporter_config, "telemetry_results", None)
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

        # Include telemetry data if available (statistical summary only, no raw hierarchy)
        telemetry_export_data = None
        if self._telemetry_results:
            self.debug(
                f"JSON export: Including telemetry data from {len(self._telemetry_results.endpoints_successful)} endpoints"
            )
            telemetry_export_data = {
                "summary": {
                    "endpoints_tested": self._telemetry_results.endpoints_tested,
                    "endpoints_successful": self._telemetry_results.endpoints_successful,
                    "start_time": datetime.fromtimestamp(
                        self._telemetry_results.start_ns / NANOS_PER_SECOND
                    ),
                    "end_time": datetime.fromtimestamp(
                        self._telemetry_results.end_ns / NANOS_PER_SECOND
                    ),
                },
                "endpoints": self._generate_telemetry_statistical_summary(),
            }

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

    def _generate_telemetry_statistical_summary(self) -> dict:
        """Generate clean statistical summary of telemetry data for JSON export."""
        summary = {}

        if not self._telemetry_results or not self._telemetry_results.telemetry_data:
            return summary

        for (
            dcgm_url,
            gpus_data,
        ) in self._telemetry_results.telemetry_data.dcgm_endpoints.items():
            endpoint_display = dcgm_url.replace("http://", "").replace("/metrics", "")
            summary[endpoint_display] = {"gpus": {}}

            # Define metrics to include in summary
            metrics_to_export = [
                ("gpu_power_usage", "W"),
                ("gpu_power_limit", "W"),
                ("energy_consumption", "MJ"),
                ("gpu_utilization", "%"),
                ("gpu_memory_used", "GB"),
                ("gpu_temperature", "°C"),
                ("sm_clock_frequency", "MHz"),
                ("memory_clock_frequency", "MHz"),
            ]

            for gpu_uuid, gpu_data in gpus_data.items():
                gpu_summary = {
                    "gpu_index": gpu_data.metadata.gpu_index,
                    "gpu_name": gpu_data.metadata.model_name,
                    "gpu_uuid": gpu_uuid,
                    "hostname": gpu_data.metadata.hostname,
                    "metrics": {},
                }

                # Add statistical summary for each metric
                for metric_key, unit in metrics_to_export:
                    try:
                        metric_result = gpu_data.get_metric_result(
                            metric_key, metric_key, metric_key, unit
                        )
                        gpu_summary["metrics"][metric_key] = {
                            "avg": round(metric_result.avg, 2)
                            if metric_result.avg is not None
                            else None,
                            "min": round(metric_result.min, 2)
                            if metric_result.min is not None
                            else None,
                            "max": round(metric_result.max, 2)
                            if metric_result.max is not None
                            else None,
                            "p99": round(metric_result.p99, 2)
                            if metric_result.p99 is not None
                            else None,
                            "p90": round(metric_result.p90, 2)
                            if metric_result.p90 is not None
                            else None,
                            "p75": round(metric_result.p75, 2)
                            if metric_result.p75 is not None
                            else None,
                            "std": round(metric_result.std, 2)
                            if metric_result.std is not None
                            else None,
                            "count": metric_result.count,
                            "unit": unit,
                        }
                    except Exception:
                        # Skip metrics without data
                        continue

                summary[endpoint_display]["gpus"][
                    f"gpu_{gpu_data.metadata.gpu_index}"
                ] = gpu_summary

        return summary
