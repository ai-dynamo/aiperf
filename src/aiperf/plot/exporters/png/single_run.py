# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-run PNG exporter for time series plots.

Generates static PNG images for analyzing a single profiling run.
"""

from pathlib import Path

import plotly.graph_objects as go

import aiperf.plot.handlers.single_run_handlers  # noqa: F401
from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.plot_specs import (
    DataSource,
    PlotSpec,
)
from aiperf.plot.exporters.png.base import BasePNGExporter
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


class SingleRunPNGExporter(BasePNGExporter):
    """
    PNG exporter for single-run time series plots.

    Generates static PNG images for analyzing a single profiling run:
    1. TTFT over time (scatter)
    2. ITL over time (scatter)
    3. Latency over time (area)
    4. Timeslice plots (histograms) - if timeslice data available:
       - TTFT across time slices
       - ITL across time slices
       - Throughput across time slices
       - Latency across time slices
    """

    def export(
        self, run: RunData, available_metrics: dict, plot_specs: list[PlotSpec]
    ) -> list[Path]:
        """
        Export single-run time series plots as PNG files.

        Args:
            run: RunData object with per-request data
            available_metrics: Dictionary with display_names and units for metrics
            plot_specs: List of plot specifications defining which plots to generate

        Returns:
            List of Path objects for generated PNG files
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        for spec in plot_specs:
            try:
                missing = self._missing_data_sources(spec, run)
                if missing:
                    self.warning(
                        f"Skipping plot '{spec.name}': required data source(s) "
                        f"missing or empty: {missing}. Verify the benchmark "
                        f"produced this data (e.g. 'gpu_telemetry' requires "
                        f"--gpu-telemetry-url)."
                    )
                    continue

                fig = self._create_plot_from_spec(spec, run, available_metrics)

                path = self.output_dir / spec.filename
                self._export_figure(fig, path)
                self.debug(f"Generated {spec.filename}")
                generated_files.append(path)

            except Exception as e:
                self.error(f"Failed to generate {spec.name}: {e}")

        self._create_summary_file(generated_files)

        return generated_files

    def _missing_data_sources(self, spec: PlotSpec, run: RunData) -> list[str]:
        """Return the data sources required by ``spec`` that are missing/empty.

        Deduplicated and sorted so the warning message is stable. Empty
        list means the plot can be generated. A spec metric counts as
        missing when its declared ``source`` resolves to ``None`` or an
        empty DataFrame on the RunData.
        """
        missing: set[str] = set()
        for metric in spec.metrics:
            if metric.source == DataSource.REQUESTS and (
                run.requests is None or run.requests.empty
            ):
                missing.add("requests")
            elif metric.source == DataSource.TIMESLICES and (
                run.timeslices is None or run.timeslices.empty
            ):
                missing.add("timeslices")
            elif metric.source == DataSource.GPU_TELEMETRY and (
                run.gpu_telemetry is None or run.gpu_telemetry.empty
            ):
                missing.add("gpu_telemetry")
        return sorted(missing)

    def _create_plot_from_spec(
        self, spec: PlotSpec, run: RunData, available_metrics: dict
    ) -> go.Figure:
        """
        Create a plot figure from a plot specification using the factory pattern.

        Args:
            spec: Plot specification
            run: RunData object
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            Plotly figure object
        """
        HandlerClass = plugins.get_class(PluginType.PLOT, spec.plot_type)
        handler = HandlerClass(
            plot_generator=self.plot_generator,
            logger=self,
        )

        return handler.create_plot(spec, run, available_metrics)
