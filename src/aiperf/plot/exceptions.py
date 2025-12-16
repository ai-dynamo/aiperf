# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom exceptions for the visualization package.

This module defines exception classes used throughout the visualization
functionality to handle various error conditions.
"""


class PlotError(Exception):
    """
    Base exception for all visualization-related errors.

    This is the parent class for all custom exceptions in the plot package.
    """

    pass


class DataLoadError(PlotError):
    """
    Exception raised when data loading fails.

    This exception is raised when there are issues loading profiling data
    files, such as missing files, corrupted data, or invalid formats.

    Args:
        message: A description of the data loading error.
        path: Optional path to the file that caused the error.
    """

    def __init__(self, message: str, path: str | None = None) -> None:
        """Initialize DataLoadError with message and optional path."""
        if path is not None:
            super().__init__(f"{message}: {path}")
        else:
            super().__init__(message)
        self.path: str | None = path


class PlotGenerationError(PlotError):
    """
    Exception raised when plot generation fails.

    This exception is raised when there are issues generating visualizations,
    such as invalid plot specifications, data processing errors, or rendering
    failures.

    Args:
        message: A description of the plot generation error.
        plot_type: Optional type of plot that failed to generate.
    """

    def __init__(self, message: str, plot_type: str | None = None) -> None:
        """Initialize PlotGenerationError with message and optional plot type."""
        if plot_type is not None:
            super().__init__(f"{message} (plot type: {plot_type})")
        else:
            super().__init__(message)
        self.plot_type: str | None = plot_type


class ModeDetectionError(PlotError):
    """
    Exception raised when visualization mode cannot be detected.

    This exception is raised when the system cannot determine whether the
    input represents a single run or multiple runs, or when the directory
    structure is ambiguous or invalid.

    Args:
        message: A description of the mode detection error.
    """

    pass


class DataUnavailableError(PlotError):
    """
    Exception raised when required data is not available for a plot.

    This exception provides helpful messages explaining why a plot cannot be
    generated and what data would be needed.

    Args:
        message: A user-friendly description of what data is missing
        data_type: The type of data that is unavailable (e.g., "timeslice", "gpu_telemetry")
        hint: Optional hint about how to generate the missing data
    """

    def __init__(
        self, message: str, data_type: str | None = None, hint: str | None = None
    ) -> None:
        """Initialize DataUnavailableError with message, data type, and hint."""
        full_message = message
        if hint is not None:
            full_message = f"{message}\n{hint}"
        super().__init__(full_message)
        self.data_type: str | None = data_type
        self.hint: str | None = hint
