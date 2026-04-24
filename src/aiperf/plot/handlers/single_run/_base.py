# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class and shared helpers for single-run plot handlers."""

import pandas as pd

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.data_preparation import prepare_request_timeseries
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import DataSource, MetricSpec
from aiperf.plot.exceptions import PlotGenerationError
from aiperf.plot.metric_names import get_gpu_metric_unit

_DISTRIBUTION_STATS: frozenset[str] = frozenset(
    {"p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "std", "min", "max"}
)


def _is_single_stat_metric(metric) -> bool:
    """
    Check if metric only has 'avg' stat (no distribution stats like p50, std, etc.).

    Single-stat metrics are derived values (like throughput, count) where the aggregated
    "avg" is a calculated value (total/duration), not a statistical average of samples.

    Args:
        metric: MetricResult object or dict containing metric data

    Returns:
        True if metric only has 'avg' stat, False otherwise
    """
    for stat in _DISTRIBUTION_STATS:
        if hasattr(metric, stat):
            val = getattr(metric, stat)
        elif isinstance(metric, dict):
            val = metric.get(stat)
        else:
            continue
        if val is not None:
            return False

    return True


class BaseSingleRunHandler:
    """
    Base class for single-run plot handlers.

    Provides common functionality for data preparation and validation.
    """

    def __init__(self, plot_generator: PlotGenerator, logger=None) -> None:
        """
        Initialize the handler.

        Args:
            plot_generator: PlotGenerator instance for rendering plots
            logger: Optional logger instance
        """
        self.plot_generator = plot_generator
        self.logger = logger

    def _get_axis_label(self, metric_spec: MetricSpec, available_metrics: dict) -> str:
        """
        Get axis label for a metric.

        Args:
            metric_spec: MetricSpec object
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted axis label
        """
        if metric_spec.name == "request_number":
            return "Request Number"
        elif metric_spec.name == "timestamp":
            return "Time (seconds)"
        elif metric_spec.name == "timestamp_s":
            return "Time (s)"
        elif metric_spec.name == "Timeslice":
            return "Timeslice (s)"
        else:
            return self._get_metric_label(
                metric_spec.name, metric_spec.stat, available_metrics
            )

    def _get_custom_or_default_label(
        self,
        custom_label: str | None,
        metric_spec: MetricSpec,
        available_metrics: dict,
    ) -> str:
        """
        Get custom axis label if provided, otherwise auto-generate.

        Args:
            custom_label: Custom label from PlotSpec (x_label or y_label)
            metric_spec: MetricSpec object for fallback generation
            available_metrics: Dictionary with display_names and units

        Returns:
            Custom label if provided, otherwise auto-generated label
        """
        if custom_label:
            return custom_label
        return self._get_axis_label(metric_spec, available_metrics)

    def _get_metric_label(
        self, metric_name: str, stat: str | None, available_metrics: dict
    ) -> str:
        """
        Get formatted metric label.

        Args:
            metric_name: Name of the metric
            stat: Statistic (e.g., "avg", "p50")
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted metric label
        """
        display_name = None
        unit = ""

        if "display_names" in available_metrics or "units" in available_metrics:
            display_name = available_metrics.get("display_names", {}).get(metric_name)
            unit = available_metrics.get("units", {}).get(metric_name, "")

        if not display_name and metric_name in available_metrics:
            display_name = available_metrics[metric_name].get(
                "display_name", metric_name
            )
            unit = available_metrics[metric_name].get("unit", "")

        if display_name:
            if stat and stat not in ["avg", "value"]:
                display_name = f"{display_name} ({stat})"
            if unit:
                return f"{display_name} ({unit})"
            return display_name

        # Fallback: Check if it's a GPU metric and get unit from GPU config
        display_name = metric_name.replace("_", " ").title()
        gpu_unit = get_gpu_metric_unit(metric_name)
        # Heuristic: metrics with "utilization" in the name are percentages
        if not gpu_unit and "utilization" in metric_name.lower():
            gpu_unit = "%"
        if stat and stat not in ["avg", "value"]:
            display_name = f"{display_name} ({stat})"
        if gpu_unit:
            return f"{display_name} ({gpu_unit})"
        return display_name

    def _prepare_data_for_source(
        self, source: DataSource, run: RunData
    ) -> pd.DataFrame:
        """
        Prepare data from a specific source.

        Args:
            source: Data source to prepare
            run: RunData object

        Returns:
            Prepared DataFrame
        """
        if source == DataSource.REQUESTS:
            return prepare_request_timeseries(run)
        elif source == DataSource.TIMESLICES:
            return run.timeslices
        elif source == DataSource.GPU_TELEMETRY:
            return run.gpu_telemetry
        else:
            raise PlotGenerationError(f"Unsupported data source: {source}")
