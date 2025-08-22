# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import aiofiles
from pydantic import BaseModel

from aiperf.common.config import UserConfig
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
        self._output_directory = exporter_config.user_config.output.artifact_directory
        self._input_config = exporter_config.user_config
        self._metric_registry = MetricRegistry
        self._file_path = self._output_directory / "profile_export_aiperf.json"

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

        export_data = JsonExportData(
            input_config=self._input_config,
            records=converted_records,
            was_cancelled=self._results.was_cancelled,
            error_summary=self._results.error_summary,
            start_time=start_time,
            end_time=end_time,
        )

        self.debug(lambda: f"Exporting data to JSON file: {export_data}")
        export_data_json = export_data.model_dump_json(indent=2, exclude_unset=True)
        async with aiofiles.open(self._file_path, "w") as f:
            await f.write(export_data_json)
