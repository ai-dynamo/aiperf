# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot specifications for configurable plot generation."""

from typing import Literal

from pydantic import Field, field_validator

from aiperf.common.config import BaseConfig
from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.models import AIPerfBaseModel
from aiperf.plugin.enums import PlotType


class Style(AIPerfBaseModel):
    """Styling configuration for a plot trace."""

    mode: str = Field(
        default="lines",
        description="Plotly visualization mode ('lines', 'markers', 'lines+markers')",
    )
    line_shape: str | None = Field(
        default=None,
        description="Line shape for the trace ('linear', 'hv' for step, 'spline', or None)",
    )
    fill: str | None = Field(
        default=None,
        description="Fill pattern for the trace ('tozeroy', 'tonexty', or None for no fill)",
    )
    line_width: int = Field(
        default=2,
        description="Width of the line in pixels",
    )
    marker_size: int = Field(
        default=8,
        description="Size of markers in pixels",
    )
    marker_opacity: float = Field(
        default=1.0,
        description="Opacity of markers (0.0 to 1.0)",
    )
    fill_opacity: float = Field(
        default=0.3,
        description="Opacity of fill area (0.0 to 1.0)",
    )


class ExperimentClassificationConfig(BaseConfig):
    """Configuration for classifying runs as baseline or treatment."""

    baselines: list[str] = Field(
        default_factory=list,
        description="List of glob patterns to match baseline runs (e.g., '*_agg_*', '*baseline*')",
    )
    treatments: list[str] = Field(
        default_factory=list,
        description="List of glob patterns to match treatment runs (e.g., '*_disagg_*', '*kvrouter*')",
    )
    default: Literal["baseline", "treatment"] = Field(
        default="treatment",
        description="Default classification when no patterns match",
    )
    group_extraction_pattern: str | None = Field(
        default=r"^(baseline|treatment_\d+)",
        description="Regex pattern to extract experiment group from run name or parent directory names. "
        "First capture group is used. Example: '^(baseline|treatment_\\d+)' extracts 'treatment_1' "
        "from 'treatment_1_large_input_small_output'. Used for grouping treatment variants.",
    )
    group_display_names: dict[str, str] | None = Field(
        default=None,
        description="Optional mapping of experiment group IDs to human-readable display names for legends. "
        "Example: {'baseline': 'Baseline', 'treatment_1': 'Large Input Small Output'}",
    )


class DataSource(CaseInsensitiveStrEnum):
    """Data sources for plot metrics."""

    REQUESTS = "requests"
    TIMESLICES = "timeslices"
    GPU_TELEMETRY = "gpu_telemetry"
    AGGREGATED = "aggregated"
    SERVER_METRICS = "server_metrics"
    SERVER_METRICS_AGGREGATED = "server_metrics_aggregated"


class MetricSpec(AIPerfBaseModel):
    """Specification for a single metric in a plot."""

    name: str = Field(description="Name of the metric (column name in DataFrame)")
    source: DataSource = Field(description="Data source where the metric comes from")
    axis: Literal["x", "y", "y2"] = Field(
        description="Which axis the metric should be plotted on"
    )
    stat: str | None = Field(
        default=None,
        description="Optional statistic to filter/extract (e.g., 'avg', 'p50', 'p95'). "
        "Applies to timeslices, aggregated data, and any source with stats",
    )


class PlotSpec(AIPerfBaseModel):
    """Base specification for a plot."""

    name: str = Field(description="Unique identifier for the plot")
    plot_type: PlotType = Field(description="Type of plot to generate")
    metrics: list[MetricSpec] = Field(description="List of metrics to plot")
    title: str | None = Field(
        default=None, description="Plot title (auto-generated if None)"
    )
    filename: str | None = Field(
        default=None, description="Output filename (auto-generated from name if None)"
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of what this plot shows",
    )
    label_by: str | None = Field(
        default=None,
        description="Column to use for labeling points (single column only). "
        "Must be provided as a single-element list in YAML (e.g., [concurrency]).",
    )
    group_by: str | None = Field(
        default=None,
        description="Column to use for grouping data into separate series (single column only). "
        "Must be provided as a single-element list in YAML (e.g., [model]). "
        "Note: When experiment_classification is enabled, this is auto-overridden to 'experiment_group'.",
    )

    @field_validator("label_by", "group_by", mode="before")
    @classmethod
    def _normalize_list_to_string(cls, v: str | list[str] | None) -> str | None:
        """Convert single-element list to string.

        Args:
            v: Single-element list, string, or None

        Returns:
            String value or None

        Raises:
            ValueError: If v is not a single-element list, string, or None
        """
        if v is None:
            return None

        if isinstance(v, str):
            return v

        if isinstance(v, list):
            if len(v) == 0:
                return None
            if len(v) == 1:
                return v[0]
            raise ValueError(
                f"Multi-column grouping is not supported. "
                f"Provide a single column as a string or single-element list, got: {v}"
            )

        raise ValueError(
            f"label_by and group_by must be a string or list, got {type(v).__name__}"
        )

    x_label: str | None = Field(
        default=None,
        description="Custom x-axis label (auto-generated from metric name if None)",
    )
    y_label: str | None = Field(
        default=None,
        description="Custom y-axis label (auto-generated from metric name if None)",
    )
    primary_style: Style | None = Field(
        default=None,
        description="Style configuration for primary (y) axis trace",
    )
    secondary_style: Style | None = Field(
        default=None,
        description="Style configuration for secondary (y2) axis trace",
    )
    supplementary_col: str | None = Field(
        default=None,
        description="Optional supplementary column name (e.g., 'active_requests')",
    )
    autoscale: Literal["none", "x", "y", "both"] = Field(
        default="none",
        description="Which axes to autoscale ('none', 'x', 'y', 'both')",
    )


class TimeSlicePlotSpec(PlotSpec):
    """Specification for timeslice scatter plots."""

    use_slice_duration: bool = Field(
        default=True,
        description="Whether to pass slice_duration to the plot generator "
        "for proper time-based x-axis formatting",
    )
