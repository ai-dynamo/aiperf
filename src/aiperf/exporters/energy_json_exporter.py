# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON exporter for energy efficiency metrics."""

from __future__ import annotations

import orjson

from aiperf.analysis.energy_analyzer import EnergyEfficiencySummary
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter


class EnergyJsonExporter(MetricsBaseExporter):
    """Exports energy efficiency metrics to a JSON file."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        if exporter_config.energy_efficiency_results is None:
            raise DataExporterDisabled("No energy efficiency results available")
        self._summary: EnergyEfficiencySummary = (
            exporter_config.energy_efficiency_results
        )
        self._file_path = (
            exporter_config.config.artifacts.profile_export_energy_efficiency_json_file
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Energy Efficiency JSON Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        return orjson.dumps(
            self._summary.to_json(), option=orjson.OPT_INDENT_2, default=str
        ).decode()
