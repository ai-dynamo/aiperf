# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for plot type handler protocol and registry integration.

Tests for PlotTypeHandlerProtocol in aiperf.plot.core.plot_type_handlers module.
"""

from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go
import pytest

from aiperf.plot.core.plot_specs import PlotSpec
from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import PlotType


@pytest.fixture
def mock_plot_generator() -> MagicMock:
    """
    Create a mock PlotGenerator instance.

    Returns:
        MagicMock instance representing PlotGenerator
    """
    return MagicMock()


@pytest.fixture
def mock_plot_spec() -> MagicMock:
    """
    Create a mock PlotSpec instance.

    Returns:
        MagicMock instance representing PlotSpec
    """
    spec = MagicMock(spec=PlotSpec)
    spec.plot_type = PlotType.SCATTER
    spec.name = "test_plot"
    return spec


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Create a sample DataFrame for testing.

    Returns:
        Sample pandas DataFrame
    """
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


class TestPlotTypeHandlerProtocol:
    """Test suite for PlotTypeHandlerProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """
        Test that PlotTypeHandlerProtocol is runtime checkable.

        Verifies that the protocol is decorated with @runtime_checkable.
        """
        assert hasattr(PlotTypeHandlerProtocol, "_is_runtime_protocol")
        assert PlotTypeHandlerProtocol._is_runtime_protocol is True

    def test_protocol_defines_init_method(self):
        """Test that protocol defines __init__ method signature."""
        assert hasattr(PlotTypeHandlerProtocol, "__init__")

    def test_protocol_defines_can_handle_method(self):
        """Test that protocol defines can_handle method."""
        assert hasattr(PlotTypeHandlerProtocol, "can_handle")

    def test_protocol_defines_create_plot_method(self):
        """Test that protocol defines create_plot method."""
        assert hasattr(PlotTypeHandlerProtocol, "create_plot")

    def test_mock_handler_satisfies_protocol(self, mock_plot_generator):
        """Test that a mock handler satisfies the protocol using isinstance()."""

        class MockHandler:
            def __init__(self, plot_generator, **kwargs):
                self.plot_generator = plot_generator

            def can_handle(
                self, spec: PlotSpec, data: pd.DataFrame | MagicMock
            ) -> bool:
                return True

            def create_plot(
                self,
                spec: PlotSpec,
                data: pd.DataFrame | MagicMock,
                available_metrics: dict,
            ) -> go.Figure:
                return go.Figure()

        handler = MockHandler(mock_plot_generator)
        assert isinstance(handler, PlotTypeHandlerProtocol)

    def test_incomplete_handler_does_not_satisfy_protocol(self, mock_plot_generator):
        """Test that incomplete handler does not satisfy protocol."""

        class IncompleteHandler:
            def __init__(self, plot_generator, **kwargs):
                self.plot_generator = plot_generator

            def can_handle(
                self, spec: PlotSpec, data: pd.DataFrame | MagicMock
            ) -> bool:
                return True

        handler = IncompleteHandler(mock_plot_generator)
        assert not isinstance(handler, PlotTypeHandlerProtocol)

    def test_handler_missing_method_does_not_satisfy_protocol(self):
        """Test that handler missing a method does not satisfy protocol."""

        class HandlerMissingCanHandle:
            def __init__(self, plot_generator, **kwargs):
                self.plot_generator = plot_generator

            def create_plot(
                self,
                spec: PlotSpec,
                data: pd.DataFrame | MagicMock,
                available_metrics: dict,
            ) -> go.Figure:
                return go.Figure()

        handler = HandlerMissingCanHandle(MagicMock())
        assert not isinstance(handler, PlotTypeHandlerProtocol)


class TestPlotTypeHandlerRegistry:
    """Test suite for plot type handler registry integration."""

    def test_plot_types_are_registered(self):
        """Test that plot types are registered in the plugin registry."""
        entries = list(plugins.iter_all("plot"))
        assert len(entries) > 0

    @pytest.mark.parametrize("plot_type", [PlotType.SCATTER, PlotType.TIMESLICE, PlotType.HISTOGRAM, PlotType.AREA])  # fmt: skip
    def test_common_plot_types_registered(self, plot_type: PlotType):
        """Test that common plot types are registered."""
        registered_names = [entry.name for entry, _ in plugins.iter_all("plot")]
        assert plot_type.value in registered_names

    def test_get_class_returns_handler(self, mock_plot_generator):
        """Test that get_class returns a valid handler class."""
        HandlerClass = plugins.get_class("plot", PlotType.SCATTER.value)
        handler = HandlerClass(plot_generator=mock_plot_generator)
        assert isinstance(handler, PlotTypeHandlerProtocol)

    def test_created_handler_has_required_methods(self, mock_plot_generator):
        """Test that created handler has required protocol methods."""
        HandlerClass = plugins.get_class("plot", PlotType.SCATTER.value)
        handler = HandlerClass(plot_generator=mock_plot_generator)
        assert hasattr(handler, "can_handle")
        assert hasattr(handler, "create_plot")
        assert callable(handler.can_handle)
        assert callable(handler.create_plot)
