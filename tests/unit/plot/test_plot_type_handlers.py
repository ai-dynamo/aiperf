# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for plot type handler factory.

Tests for factory pattern in aiperf.plot.core.plot_type_handlers module.
"""

from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go
import pytest

from aiperf.common.enums import EndpointType
from aiperf.common.factories import EndpointFactory
from aiperf.plot.core.plot_specs import PlotSpec, PlotType
from aiperf.plot.core.plot_type_handlers import (
    PlotTypeHandlerFactory,
    PlotTypeHandlerProtocol,
)


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


@pytest.fixture
def mock_run_data() -> MagicMock:
    """
    Create a mock RunData instance.

    Returns:
        MagicMock instance representing RunData
    """
    return MagicMock()


@pytest.fixture
def available_metrics() -> dict:
    """
    Create a sample available_metrics dictionary.

    Returns:
        Dictionary with display_names and units for metrics
    """
    return {
        "display_names": {"x": "X Axis", "y": "Y Axis"},
        "units": {"x": "units", "y": "values"},
    }


@pytest.fixture(autouse=True)
def save_and_restore_registry():
    """
    Save and restore the factory registry to avoid test pollution.

    This fixture ensures that any registrations made during tests
    are cleaned up afterwards.
    """
    original_registry = PlotTypeHandlerFactory._registry.copy()
    original_priorities = PlotTypeHandlerFactory._override_priorities.copy()
    yield
    PlotTypeHandlerFactory._registry = original_registry
    PlotTypeHandlerFactory._override_priorities = original_priorities


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


class TestPlotTypeHandlerFactory:
    """Test suite for PlotTypeHandlerFactory."""

    def test_factory_has_registry_attribute(self):
        """Test that factory has _registry attribute."""
        assert hasattr(PlotTypeHandlerFactory, "_registry")
        assert isinstance(PlotTypeHandlerFactory._registry, dict)

    def test_factory_has_override_priorities_attribute(self):
        """Test that factory has _override_priorities attribute."""
        assert hasattr(PlotTypeHandlerFactory, "_override_priorities")
        assert isinstance(PlotTypeHandlerFactory._override_priorities, dict)

    def test_factory_has_logger_attribute(self):
        """Test that factory has _logger attribute."""
        assert hasattr(PlotTypeHandlerFactory, "_logger")

    def test_register_decorator_adds_handler_to_registry(self, mock_plot_generator):
        """Test that register decorator adds handler to registry."""

        @PlotTypeHandlerFactory.register(PlotType.SCATTER, override_priority=100)
        class TestScatterHandler:
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

        assert PlotType.SCATTER in PlotTypeHandlerFactory._registry
        assert PlotTypeHandlerFactory._registry[PlotType.SCATTER] == TestScatterHandler

    def test_register_with_override_priority(self, mock_plot_generator):
        """Test registering handler with override priority."""

        @PlotTypeHandlerFactory.register(PlotType.AREA, override_priority=10)
        class HighPriorityHandler:
            def __init__(self, plot_generator, **kwargs):
                pass

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

        assert PlotType.AREA in PlotTypeHandlerFactory._registry
        assert PlotTypeHandlerFactory._override_priorities[PlotType.AREA] == 10

    def test_register_with_higher_priority_overrides_existing(
        self, mock_plot_generator
    ):
        """Test that handler with higher priority overrides existing handler."""

        @PlotTypeHandlerFactory.register(PlotType.HISTOGRAM, override_priority=0)
        class LowPriorityHandler:
            def __init__(self, plot_generator, **kwargs):
                pass

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

        @PlotTypeHandlerFactory.register(PlotType.HISTOGRAM, override_priority=5)
        class HighPriorityHandler:
            def __init__(self, plot_generator, **kwargs):
                pass

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

        assert (
            PlotTypeHandlerFactory._registry[PlotType.HISTOGRAM] == HighPriorityHandler
        )
        assert PlotTypeHandlerFactory._override_priorities[PlotType.HISTOGRAM] == 5

    def test_register_with_same_priority_does_not_override(self, mock_plot_generator):
        """Test that handler with same priority does not override existing handler."""

        @PlotTypeHandlerFactory.register(PlotType.TIMESLICE, override_priority=5)
        class FirstHandler:
            def __init__(self, plot_generator, **kwargs):
                pass

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

        @PlotTypeHandlerFactory.register(PlotType.TIMESLICE, override_priority=5)
        class SecondHandler:
            def __init__(self, plot_generator, **kwargs):
                pass

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

        assert PlotTypeHandlerFactory._registry[PlotType.TIMESLICE] == FirstHandler
        assert PlotTypeHandlerFactory._override_priorities[PlotType.TIMESLICE] == 5

    def test_create_instance_creates_handler_with_kwargs(self, mock_plot_generator):
        """Test that create_instance creates handler instances with kwargs."""

        @PlotTypeHandlerFactory.register(PlotType.PARETO, override_priority=100)
        class ParetoHandler:
            def __init__(self, plot_generator, custom_arg=None, **kwargs):
                self.plot_generator = plot_generator
                self.custom_arg = custom_arg

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

        handler = PlotTypeHandlerFactory.create_instance(
            PlotType.PARETO,
            plot_generator=mock_plot_generator,
            custom_arg="test_value",
        )

        assert isinstance(handler, ParetoHandler)
        assert handler.plot_generator == mock_plot_generator
        assert handler.custom_arg == "test_value"

    def test_get_all_class_types_returns_list(self):
        """Test that get_all_class_types returns list of registered types."""

        @PlotTypeHandlerFactory.register(PlotType.SCATTER_LINE)
        class ScatterLineHandler:
            def __init__(self, plot_generator, **kwargs):
                pass

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

        types = PlotTypeHandlerFactory.get_all_class_types()
        assert isinstance(types, list)
        assert PlotType.SCATTER_LINE in types

    def test_get_all_classes_returns_list_of_handler_classes(self):
        """Test that get_all_classes returns list of handler classes."""

        @PlotTypeHandlerFactory.register(PlotType.DUAL_AXIS, override_priority=100)
        class DualAxisHandler:
            def __init__(self, plot_generator, **kwargs):
                pass

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

        classes = PlotTypeHandlerFactory.get_all_classes()
        assert isinstance(classes, list)
        assert DualAxisHandler in classes

    def test_get_class_from_type_returns_handler_class(self):
        """Test that get_class_from_type returns the handler class."""

        @PlotTypeHandlerFactory.register(
            PlotType.SCATTER_WITH_PERCENTILES, override_priority=100
        )
        class ScatterWithPercentilesHandler:
            def __init__(self, plot_generator, **kwargs):
                pass

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

        handler_class = PlotTypeHandlerFactory.get_class_from_type(
            PlotType.SCATTER_WITH_PERCENTILES
        )
        assert handler_class == ScatterWithPercentilesHandler

    def test_get_all_classes_and_types_returns_tuples(self):
        """Test that get_all_classes_and_types returns list of (class, type) tuples."""

        @PlotTypeHandlerFactory.register(
            PlotType.REQUEST_TIMELINE, override_priority=100
        )
        class RequestTimelineHandler:
            def __init__(self, plot_generator, **kwargs):
                pass

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

        classes_and_types = PlotTypeHandlerFactory.get_all_classes_and_types()
        assert isinstance(classes_and_types, list)
        assert any(
            cls == RequestTimelineHandler and type_enum == PlotType.REQUEST_TIMELINE
            for cls, type_enum in classes_and_types
        )

    def test_register_returns_original_class(self):
        """Test that register decorator returns the original class unchanged."""

        class OriginalHandler:
            def __init__(self, plot_generator, **kwargs):
                pass

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

        decorated_class = PlotTypeHandlerFactory.register(PlotType.AREA)(
            OriginalHandler
        )
        assert decorated_class is OriginalHandler

    def test_factory_registry_isolation(self):
        """Test that each factory subclass has its own registry."""
        assert len(PlotTypeHandlerFactory._registry) > 0
        assert len(EndpointFactory._registry) > 0
        assert PlotTypeHandlerFactory._registry != EndpointFactory._registry
        assert all(
            isinstance(plot_type, PlotType)
            for plot_type in PlotTypeHandlerFactory._registry
        )
        assert all(
            isinstance(endpoint_type, EndpointType)
            for endpoint_type in EndpointFactory._registry
        )
