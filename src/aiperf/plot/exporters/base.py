# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base exporter class for all visualization formats.

Provides shared functionality for metric labeling and plot generation
across PNG, HTML, and Dash exporters.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin
from aiperf.plot.constants import PlotTheme
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.metric_names import get_metric_display_name


class BaseExporter(AIPerfLoggerMixin, ABC):
    """
    Base class for all plot exporters (PNG, HTML, Dash).

    Provides shared functionality like metric labeling and plot generation.
    Subclasses implement format-specific export logic.
    """

    def __init__(
        self,
        output_dir: Path,
        theme: PlotTheme = PlotTheme.LIGHT,
        color_pool_size: int = 10,
    ) -> None:
        """
        Initialize the base exporter.

        Args:
            output_dir: Directory where exported files will be saved
            theme: Theme to use for plots (LIGHT or DARK). Defaults to LIGHT.
            color_pool_size: Number of colors to pre-generate for group assignments.
                Defaults to 10 (seaborn's perceptual limit for distinct colors).
                Colors cycle for plots with more groups.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.theme = theme
        self.plot_generator = PlotGenerator(
            theme=theme, color_pool_size=color_pool_size
        )

    @abstractmethod
    def export(self, *args: Any, **kwargs: Any) -> list[Path]:
        """
        Export plots. Must be implemented by subclasses.

        Returns:
            List of generated file paths
        """
        raise NotImplementedError

    def _get_metric_label(
        self, metric_tag: str, stat: str | None, available_metrics: dict
    ) -> str:
        """
        Get formatted label for a metric including unit.

        Args:
            metric_tag: Metric identifier (e.g., "time_to_first_token")
            stat: Statistical measure (e.g., "p50", "avg"), or None
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted label string with stat and unit

        Examples:
            >>> self._get_metric_label("request_latency", "p50", metrics)
            'Request Latency P50 (ms)'
        """
        display_names = available_metrics.get("display_names", {})
        units = available_metrics.get("units", {})

        name = display_names.get(metric_tag, get_metric_display_name(metric_tag))
        unit = units.get(metric_tag, "")

        if not unit and metric_tag in available_metrics:
            unit = available_metrics[metric_tag].get("unit", "")

        label = f"{name} {stat.upper()}" if stat and stat != "value" else name

        if unit:
            label += f" ({unit})"

        return label
