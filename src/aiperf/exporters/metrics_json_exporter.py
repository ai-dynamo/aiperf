# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from datetime import datetime

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import (
    JsonExportData,
    JsonMetricResult,
    RunInfo,
)
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter


class MetricsJsonExporter(MetricsBaseExporter):
    """
    A class to export records to a JSON file.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        self._file_path = exporter_config.config.artifacts.profile_export_json_file
        self._run = exporter_config.run
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
        prepared_json_metrics = self._prepare_metrics_for_json(
            self._results.records or []
        )

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

        from aiperf import __version__ as aiperf_version

        # Note: server_metrics_data is exported to a separate file via ServerMetricsJsonExporter
        export_data = JsonExportData(
            schema_version=JsonExportData.SCHEMA_VERSION,
            aiperf_version=aiperf_version,
            benchmark_id=self._run.benchmark_id
            if self._run is not None
            else self._config.artifacts.benchmark_id,
            input_config=self._config,
            run_info=RunInfo.from_run(self._run),
            was_cancelled=self._results.was_cancelled,
            error_summary=self._results.error_summary,
            start_time=start_time,
            end_time=end_time,
            telemetry_data=self._telemetry_results,
        )

        # Add all prepared metrics dynamically
        for metric_tag, json_result in prepared_json_metrics.items():
            setattr(export_data, metric_tag, json_result)

        # Multi-turn TTFT trend: per-``turn_index`` MetricResult dict from
        # ``MetricsAccumulator.summarize()``. Surfaced as a top-level
        # ``multi_turn_ttft_trend`` key keyed by turn-index string so the
        # JSON shape is dict[str, JsonMetricResult] — distinguishable from
        # the flat per-tag percentile dicts at the same level. Only populated
        # when records carry ``turn_index`` metadata. ``getattr`` because
        # test doubles may not declare the attribute.
        trend = getattr(self._results, "multi_turn_ttft_trend", None)
        if trend:
            export_data.multi_turn_ttft_trend = {
                str(turn): mr.to_json_result() for turn, mr in sorted(trend.items())
            }

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
