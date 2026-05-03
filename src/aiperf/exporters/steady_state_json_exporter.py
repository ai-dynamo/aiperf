# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON exporter for steady-state windowed metrics."""

from __future__ import annotations

from typing import Any

import orjson

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.post_processors.steady_state_analyzer import SteadyStateSummary


class SteadyStateJsonExporter(MetricsBaseExporter):
    """Exports steady-state windowed metrics to a JSON file."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        if exporter_config.steady_state_results is None:
            raise DataExporterDisabled("No steady-state results available")
        self._summary: SteadyStateSummary = exporter_config.steady_state_results
        self._file_path = (
            exporter_config.config.artifacts.profile_export_steady_state_json_file
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Steady-State JSON Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        prepared = self._prepare_metrics(self._summary.results.values())
        prepared.update(self._summary.sweep_metrics)

        data: dict[str, Any] = {
            "window_metadata": self._summary.window_metadata.to_dict(),
            "metrics": {},
        }

        for tag, metric in prepared.items():
            metric_data: dict[str, Any] = {"unit": metric.unit}
            for attr in ["avg", "min", "max", "p50", "p90", "p95", "p99", "std"]:
                val = getattr(metric, attr, None)
                if val is not None:
                    metric_data[attr] = val
            data["metrics"][tag] = metric_data

        return orjson.dumps(data, option=orjson.OPT_INDENT_2, default=str).decode()
