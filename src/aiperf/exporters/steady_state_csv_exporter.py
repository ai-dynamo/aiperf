# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSV exporter for steady-state windowed metrics."""

from __future__ import annotations

import csv
import io
import numbers
from typing import Any

from aiperf.common.constants import STAT_KEYS
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.post_processors.steady_state_analyzer import SteadyStateSummary


def _format_value(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, numbers.Integral):
        return str(val)
    if isinstance(val, numbers.Real):
        return f"{val:.2f}"
    return str(val)


class SteadyStateCsvExporter(MetricsBaseExporter):
    """Exports steady-state windowed metrics to a CSV file."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(exporter_config, **kwargs)
        if exporter_config.steady_state_results is None:
            raise DataExporterDisabled("No steady-state results available")
        self._summary: SteadyStateSummary = exporter_config.steady_state_results
        self._file_path = (
            exporter_config.config.artifacts.profile_export_steady_state_csv_file
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Steady-State CSV Export",
            file_path=self._file_path,
        )

    def _write_window_metadata(self, writer: csv.writer) -> None:
        meta = self._summary.window_metadata
        writer.writerow(["# Steady-State Window Metadata"])
        writer.writerow(["detection_method", meta.detection_method])
        writer.writerow(["ramp_up_end_ns", f"{meta.ramp_up_end_ns:.0f}"])
        writer.writerow(["ramp_down_start_ns", f"{meta.ramp_down_start_ns:.0f}"])
        writer.writerow(
            ["steady_state_duration_ns", f"{meta.steady_state_duration_ns:.0f}"]
        )
        writer.writerow(["total_requests", meta.total_requests])
        writer.writerow(["steady_state_requests", meta.steady_state_requests])
        writer.writerow(["fraction_retained", f"{meta.fraction_retained:.4f}"])
        writer.writerow(
            [
                "trend_correlation",
                f"{meta.trend_correlation:.4f}"
                if meta.trend_correlation is not None
                else "",
            ]
        )
        writer.writerow(
            [
                "trend_p_value",
                f"{meta.trend_p_value:.4f}" if meta.trend_p_value is not None else "",
            ]
        )
        writer.writerow(["stationarity_warning", meta.stationarity_warning])
        writer.writerow(
            ["variance_inflation_factor", f"{meta.variance_inflation_factor:.4f}"]
        )
        writer.writerow(["effective_p99_sample_size", meta.effective_p99_sample_size])
        writer.writerow(["sample_size_warning", meta.sample_size_warning])
        if meta.bootstrap_n_iterations is not None:
            self._write_bootstrap_rows(writer, meta)
        writer.writerow([])

    @staticmethod
    def _write_bootstrap_rows(writer: csv.writer, meta: Any) -> None:
        writer.writerow(["bootstrap_n_iterations", meta.bootstrap_n_iterations])
        writer.writerow(["bootstrap_ci_ramp_up_ns", meta.bootstrap_ci_ramp_up_ns])
        writer.writerow(["bootstrap_ci_ramp_down_ns", meta.bootstrap_ci_ramp_down_ns])
        writer.writerow(["bootstrap_ci_mean_latency", meta.bootstrap_ci_mean_latency])
        writer.writerow(["bootstrap_ci_p99_latency", meta.bootstrap_ci_p99_latency])

    @staticmethod
    def _write_request_metrics(
        writer: csv.writer, request_metrics: list[MetricResult]
    ) -> None:
        writer.writerow(["Metric", *STAT_KEYS])
        for metric in request_metrics:
            row = [f"{metric.header} ({metric.unit})"]
            row.extend(_format_value(getattr(metric, key, None)) for key in STAT_KEYS)
            writer.writerow(row)

    @staticmethod
    def _write_system_metrics(
        writer: csv.writer, system_metrics: list[MetricResult]
    ) -> None:
        writer.writerow([])
        writer.writerow(["Metric", "Value"])
        for metric in system_metrics:
            writer.writerow(
                [f"{metric.header} ({metric.unit})", _format_value(metric.avg)]
            )

    def _generate_content(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)

        self._write_window_metadata(writer)

        prepared = self._prepare_metrics(self._summary.results.values())
        prepared.update(self._summary.sweep_metrics)
        if not prepared:
            return buf.getvalue()

        request_metrics = [m for m in prepared.values() if m.p50 is not None]
        system_metrics = [m for m in prepared.values() if m.p50 is None]

        if request_metrics:
            self._write_request_metrics(writer, request_metrics)
        if system_metrics:
            self._write_system_metrics(writer, system_metrics)

        return buf.getvalue()
