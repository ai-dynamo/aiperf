# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot type handlers using factory pattern for extensible plot creation.

This module provides a factory-based approach to handling different plot types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.core.plot_specs import PlotSpec

if TYPE_CHECKING:
    from aiperf.plot.core.data_loader import RunData
    from aiperf.plot.core.plot_generator import PlotGenerator


@runtime_checkable
class PlotTypeHandlerProtocol(Protocol):
    """
    Protocol for plot type handlers.

    Each handler is responsible for creating a specific type of plot
    by preparing data and delegating to PlotGenerator.
    """

    def __init__(
        self,
        plot_generator: PlotGenerator,
        **kwargs,
    ) -> None:
        """
        Initialize the plot type handler.

        Args:
            plot_generator: PlotGenerator instance for rendering plots
            **kwargs: Additional handler-specific configuration
        """
        ...

    def can_handle(self, spec: PlotSpec, data: pd.DataFrame | RunData) -> bool:
        """
        Check if this handler can generate the plot based on data availability.

        Args:
            spec: Plot specification
            data: Either a DataFrame (multi-run) or RunData (single-run)

        Returns:
            True if the plot can be generated, False otherwise
        """
        ...

    def create_plot(
        self,
        spec: PlotSpec,
        data: pd.DataFrame | RunData,
        available_metrics: dict,
    ) -> go.Figure:
        """
        Create a plot from the specification.

        Args:
            spec: Plot specification
            data: Either a DataFrame (multi-run) or RunData (single-run)
            available_metrics: Dictionary with display_names and units for metrics

        Returns:
            Plotly figure object
        """
        ...
