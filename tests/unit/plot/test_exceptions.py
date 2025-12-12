# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for plot exceptions module.

Tests for custom exception classes used in plot generation and data handling.
"""

import pytest

from aiperf.plot.exceptions import (
    DataLoadError,
    DataUnavailableError,
    ModeDetectionError,
    PlotError,
    PlotGenerationError,
)


class TestPlotError:
    """Tests for PlotError base exception."""

    def test_inherits_from_exception(self) -> None:
        """Test PlotError inherits from Exception."""
        assert issubclass(PlotError, Exception)

    def test_can_be_raised(self) -> None:
        """Test PlotError can be raised."""
        with pytest.raises(PlotError):
            raise PlotError("test error")

    def test_can_be_caught(self) -> None:
        """Test PlotError can be caught."""
        try:
            raise PlotError("test error")
        except PlotError as e:
            assert str(e) == "test error"

    def test_stores_message(self) -> None:
        """Test PlotError stores the error message."""
        error = PlotError("test message")
        assert str(error) == "test message"

    def test_empty_message(self) -> None:
        """Test PlotError with empty message."""
        error = PlotError()
        assert str(error) == ""

    def test_catches_child_exceptions(self) -> None:
        """Test PlotError catches all child exception types."""
        with pytest.raises(PlotError):
            raise DataLoadError("child exception")


class TestDataLoadError:
    """Tests for DataLoadError exception."""

    def test_inherits_from_plot_error(self) -> None:
        """Test DataLoadError inherits from PlotError."""
        assert issubclass(DataLoadError, PlotError)

    def test_message_without_path(self) -> None:
        """Test DataLoadError message without path."""
        error = DataLoadError("failed to load data")
        assert str(error) == "failed to load data"

    def test_message_with_path_includes_path(self) -> None:
        """Test DataLoadError message with path includes the path."""
        error = DataLoadError("failed to load data", path="/path/to/file.json")
        assert str(error) == "failed to load data: /path/to/file.json"
        assert "/path/to/file.json" in str(error)

    def test_stores_path_attribute(self) -> None:
        """Test DataLoadError stores path attribute."""
        error = DataLoadError("failed to load data", path="/path/to/file.json")
        assert error.path == "/path/to/file.json"

    def test_stores_none_path_when_not_provided(self) -> None:
        """Test DataLoadError stores None for path when not provided."""
        error = DataLoadError("failed to load data")
        assert error.path is None

    def test_can_be_raised(self) -> None:
        """Test DataLoadError can be raised."""
        with pytest.raises(DataLoadError):
            raise DataLoadError("test error")

    def test_can_be_caught_as_plot_error(self) -> None:
        """Test DataLoadError can be caught as PlotError."""
        with pytest.raises(PlotError):
            raise DataLoadError("test error")

    def test_path_with_special_characters(self) -> None:
        """Test DataLoadError handles paths with special characters."""
        path = "/path/with spaces/file (1).json"
        error = DataLoadError("failed", path=path)
        assert path in str(error)
        assert error.path == path


class TestPlotGenerationError:
    """Tests for PlotGenerationError exception."""

    def test_inherits_from_plot_error(self) -> None:
        """Test PlotGenerationError inherits from PlotError."""
        assert issubclass(PlotGenerationError, PlotError)

    def test_message_without_plot_type(self) -> None:
        """Test PlotGenerationError message without plot_type."""
        error = PlotGenerationError("failed to generate plot")
        assert str(error) == "failed to generate plot"

    def test_message_with_plot_type_includes_type(self) -> None:
        """Test PlotGenerationError message with plot_type includes the type."""
        error = PlotGenerationError("failed to generate plot", plot_type="scatter")
        assert str(error) == "failed to generate plot (plot type: scatter)"
        assert "scatter" in str(error)

    def test_stores_plot_type_attribute(self) -> None:
        """Test PlotGenerationError stores plot_type attribute."""
        error = PlotGenerationError("failed to generate plot", plot_type="scatter")
        assert error.plot_type == "scatter"

    def test_stores_none_plot_type_when_not_provided(self) -> None:
        """Test PlotGenerationError stores None for plot_type when not provided."""
        error = PlotGenerationError("failed to generate plot")
        assert error.plot_type is None

    def test_can_be_raised(self) -> None:
        """Test PlotGenerationError can be raised."""
        with pytest.raises(PlotGenerationError):
            raise PlotGenerationError("test error")

    def test_can_be_caught_as_plot_error(self) -> None:
        """Test PlotGenerationError can be caught as PlotError."""
        with pytest.raises(PlotError):
            raise PlotGenerationError("test error")

    def test_plot_type_with_special_characters(self) -> None:
        """Test PlotGenerationError handles plot types with special characters."""
        plot_type = "time-series-gpu"
        error = PlotGenerationError("failed", plot_type=plot_type)
        assert plot_type in str(error)
        assert error.plot_type == plot_type


class TestModeDetectionError:
    """Tests for ModeDetectionError exception."""

    def test_inherits_from_plot_error(self) -> None:
        """Test ModeDetectionError inherits from PlotError."""
        assert issubclass(ModeDetectionError, PlotError)

    def test_basic_instantiation(self) -> None:
        """Test ModeDetectionError basic instantiation."""
        error = ModeDetectionError("cannot detect mode")
        assert str(error) == "cannot detect mode"

    def test_can_be_raised(self) -> None:
        """Test ModeDetectionError can be raised."""
        with pytest.raises(ModeDetectionError):
            raise ModeDetectionError("test error")

    def test_can_be_caught_as_plot_error(self) -> None:
        """Test ModeDetectionError can be caught as PlotError."""
        with pytest.raises(PlotError):
            raise ModeDetectionError("test error")

    def test_stores_message(self) -> None:
        """Test ModeDetectionError stores the error message."""
        error = ModeDetectionError("ambiguous directory structure")
        assert "ambiguous" in str(error)

    def test_descriptive_message(self) -> None:
        """Test ModeDetectionError with descriptive message."""
        message = "Cannot determine if input is single run or multi-run"
        error = ModeDetectionError(message)
        assert str(error) == message


class TestDataUnavailableError:
    """Tests for DataUnavailableError exception."""

    def test_inherits_from_plot_error(self) -> None:
        """Test DataUnavailableError inherits from PlotError."""
        assert issubclass(DataUnavailableError, PlotError)

    def test_message_without_hint(self) -> None:
        """Test DataUnavailableError message without hint."""
        error = DataUnavailableError("timeslice data not available")
        assert str(error) == "timeslice data not available"

    def test_message_with_hint_includes_hint_on_newline(self) -> None:
        """Test DataUnavailableError message with hint includes hint on newline."""
        error = DataUnavailableError(
            "timeslice data not available",
            hint="Run with --export-timeslices to generate this data",
        )
        assert "timeslice data not available" in str(error)
        assert "Run with --export-timeslices" in str(error)
        assert "\n" in str(error)

    def test_stores_data_type_attribute(self) -> None:
        """Test DataUnavailableError stores data_type attribute."""
        error = DataUnavailableError(
            "data not available", data_type="timeslice", hint="run with flag"
        )
        assert error.data_type == "timeslice"

    def test_stores_hint_attribute(self) -> None:
        """Test DataUnavailableError stores hint attribute."""
        hint = "Run with --export-timeslices"
        error = DataUnavailableError("data not available", hint=hint)
        assert error.hint == hint

    def test_stores_none_data_type_when_not_provided(self) -> None:
        """Test DataUnavailableError stores None for data_type when not provided."""
        error = DataUnavailableError("data not available")
        assert error.data_type is None

    def test_stores_none_hint_when_not_provided(self) -> None:
        """Test DataUnavailableError stores None for hint when not provided."""
        error = DataUnavailableError("data not available")
        assert error.hint is None

    def test_can_be_raised(self) -> None:
        """Test DataUnavailableError can be raised."""
        with pytest.raises(DataUnavailableError):
            raise DataUnavailableError("test error")

    def test_can_be_caught_as_plot_error(self) -> None:
        """Test DataUnavailableError can be caught as PlotError."""
        with pytest.raises(PlotError):
            raise DataUnavailableError("test error")

    def test_with_all_parameters(self) -> None:
        """Test DataUnavailableError with all parameters."""
        error = DataUnavailableError(
            message="GPU telemetry data not available",
            data_type="gpu_telemetry",
            hint="Enable GPU telemetry collection during profiling",
        )
        assert error.data_type == "gpu_telemetry"
        assert error.hint == "Enable GPU telemetry collection during profiling"
        assert "GPU telemetry data not available" in str(error)
        assert "Enable GPU telemetry collection during profiling" in str(error)

    def test_hint_formats_with_newline(self) -> None:
        """Test DataUnavailableError hint is formatted on a new line."""
        error = DataUnavailableError(
            "Data missing",
            hint="Try this solution",
        )
        message = str(error)
        assert message == "Data missing\nTry this solution"

    def test_multiline_hint(self) -> None:
        """Test DataUnavailableError with multiline hint."""
        hint = "Try these steps:\n1. Enable flag\n2. Rerun profiling"
        error = DataUnavailableError("Data missing", hint=hint)
        assert hint in str(error)
        assert error.hint == hint


class TestExceptionHierarchy:
    """Tests for exception hierarchy relationships."""

    def test_all_custom_exceptions_inherit_from_plot_error(self) -> None:
        """Test all custom exceptions inherit from PlotError."""
        assert issubclass(DataLoadError, PlotError)
        assert issubclass(PlotGenerationError, PlotError)
        assert issubclass(ModeDetectionError, PlotError)
        assert issubclass(DataUnavailableError, PlotError)

    def test_all_custom_exceptions_inherit_from_exception(self) -> None:
        """Test all custom exceptions inherit from Exception."""
        assert issubclass(PlotError, Exception)
        assert issubclass(DataLoadError, Exception)
        assert issubclass(PlotGenerationError, Exception)
        assert issubclass(ModeDetectionError, Exception)
        assert issubclass(DataUnavailableError, Exception)

    def test_can_catch_all_with_plot_error(self) -> None:
        """Test all custom exceptions can be caught with PlotError."""
        exception_types = [
            DataLoadError("test"),
            PlotGenerationError("test"),
            ModeDetectionError("test"),
            DataUnavailableError("test"),
        ]

        for exc in exception_types:
            with pytest.raises(PlotError):
                raise exc

    def test_specific_exceptions_are_distinct(self) -> None:
        """Test specific exception types are distinct from each other."""
        assert not issubclass(DataLoadError, PlotGenerationError)
        assert not issubclass(PlotGenerationError, ModeDetectionError)
        assert not issubclass(ModeDetectionError, DataUnavailableError)
        assert not issubclass(DataUnavailableError, DataLoadError)
