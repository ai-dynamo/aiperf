# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.models import ErrorDetails, TelemetryRecord

if TYPE_CHECKING:
    from aiperf.common.models import (
        ErrorDetailsCount,
        MetricResult,
        TelemetryExportData,
    )


@runtime_checkable
class GPUTelemetryCollectorProtocol(Protocol):
    """Protocol for GPU telemetry collectors.

    Defines the interface for collectors that gather GPU metrics from various sources
    (DCGM HTTP endpoints, pynvml library, etc.) and deliver them via callbacks.
    """

    @property
    def id(self) -> str:
        """Get the collector's unique identifier."""
        ...

    @property
    def endpoint_url(self) -> str:
        """Get the source identifier (URL for DCGM, 'pynvml://localhost' for pynvml)."""
        ...

    async def initialize(self) -> None:
        """Initialize the collector resources."""
        ...

    async def start(self) -> None:
        """Start the background collection task."""
        ...

    async def stop(self) -> None:
        """Stop the collector and clean up resources."""
        ...

    async def is_url_reachable(self) -> bool:
        """Check if the collector source is available.

        For DCGM: Tests HTTP endpoint reachability.
        For pynvml: Tests NVML library initialization.

        Returns:
            True if the source is available and ready for collection.
        """
        ...


# Type aliases for callbacks
TRecordCallback = Callable[[list[TelemetryRecord], str], Awaitable[None]]
TErrorCallback = Callable[[ErrorDetails, str], Awaitable[None]]


@runtime_checkable
class GPUTelemetryProcessorProtocol(Protocol):
    """Protocol for GPU telemetry results processors that handle TelemetryRecord objects.

    This protocol is separate from ResultsProcessorProtocol because GPU telemetry data
    has fundamentally different structure (hierarchical with metadata) compared
    to inference metrics (flat key-value pairs).
    """

    async def process_telemetry_record(self, record: TelemetryRecord) -> None:
        """Process individual telemetry record with rich metadata.

        Args:
            record: TelemetryRecord containing GPU metrics and hierarchical metadata
        """
        ...


@runtime_checkable
class GPUTelemetryAccumulatorProtocol(GPUTelemetryProcessorProtocol, Protocol):
    """Protocol for GPU telemetry accumulators that accumulate GPU telemetry data and export pre-computed metrics.

    Extends GPUTelemetryProcessorProtocol to provide result export, realtime telemetry, and summarization
    capabilities. Implementations should accumulate DCGM metrics, compute aggregated statistics per GPU,
    and support dynamic dashboard enablement for realtime monitoring.
    """

    def export_results(
        self,
        start_ns: int,
        end_ns: int,
        error_summary: list[ErrorDetailsCount] | None = None,
    ) -> TelemetryExportData | None:
        """Export accumulated telemetry data as a TelemetryExportData object.

        Args:
            start_ns: Start time of collection in nanoseconds
            end_ns: End time of collection in nanoseconds
            error_summary: Optional list of error counts

        Returns:
            TelemetryExportData object with pre-computed metrics for each GPU
        """
        ...

    def start_realtime_telemetry(self) -> None:
        """Start the realtime telemetry background task.

        This is called when the user dynamically enables the telemetry dashboard
        by pressing the telemetry option in the UI without having passed the 'dashboard' parameter
        at startup.
        """

    async def summarize(self) -> list[MetricResult]:
        """Generate MetricResult list with hierarchical tags for telemetry data.

        Returns:
            List of MetricResult objects with hierarchical tags that preserve
            dcgm_url -> gpu_uuid grouping structure for dashboard filtering.
        """
        ...
