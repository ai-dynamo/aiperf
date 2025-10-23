# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from datetime import datetime

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import MetricFlags
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import (
    EndpointData,
    GpuSummary,
    JsonExportData,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.common.models.telemetry_models import TelemetryResults
from aiperf.common.types import MetricTagT
from aiperf.exporters.display_units_utils import (
    convert_all_metrics_to_display_units,
    normalize_endpoint_display,
)
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_METRICS_CONFIG
from aiperf.metrics.metric_registry import MetricRegistry


class BaseJsonExporter(AIPerfLoggerMixin):
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._telemetry_results = exporter_config.telemetry_results
        self._input_config = exporter_config.user_config
        self._metric_registry = MetricRegistry
        self._output_directory = exporter_config.user_config.output.artifact_directory
        self._file_path = exporter_config.user_config.output.profile_export_json_file

    def _generate_json_content(
        self,
        metric_results: Mapping[str, MetricResult],
        telemetry_results: TelemetryResults | None = None,
    ) -> str:
        """Generate JSON content string from inference and telemetry data.

        Args:
            records: Mapping of metric tags to MetricResult objects (inference metrics)
            telemetry_results: Optional GPU telemetry data to include

        Returns:
            str: Complete JSON content with all sections formatted and ready to write
        """

        converted_metric_results = convert_all_metrics_to_display_units(
            metric_results, self._metric_registry
        )

        filtered_metric_results = {
            k: v for k, v in converted_metric_results.items() if self._should_export(v)
        }

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
        if telemetry_results:
            summary = TelemetrySummary(
                endpoints_configured=telemetry_results.endpoints_configured,
                endpoints_successful=telemetry_results.endpoints_successful,
                start_time=datetime.fromtimestamp(
                    telemetry_results.start_ns / NANOS_PER_SECOND
                ),
                end_time=datetime.fromtimestamp(
                    telemetry_results.end_ns / NANOS_PER_SECOND
                ),
            )
            telemetry_export_data = TelemetryExportData(
                summary=summary,
                endpoints=self._generate_telemetry_statistical_summary(),
            )

        export_data = JsonExportData(
            input_config=self._input_config,
            was_cancelled=self._results.was_cancelled,
            error_summary=self._results.error_summary,
            start_time=start_time,
            end_time=end_time,
            telemetry_data=telemetry_export_data,
        )
        for metric, result in filtered_metric_results.items():
            setattr(export_data, metric, result.to_json_result())

        self.debug(lambda: f"Exporting data to JSON file: {export_data}")
        return export_data.model_dump_json(indent=2, exclude_unset=True)

    def _should_export(self, metric: MetricResult) -> bool:
        """Check if a metric should be exported."""
        metric_class = MetricRegistry.get_class(metric.tag)
        res = metric_class.missing_flags(
            MetricFlags.EXPERIMENTAL | MetricFlags.INTERNAL
        )
        self.debug(lambda: f"Metric '{metric.tag}' should be exported: {res}")
        return res

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
