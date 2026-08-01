# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiperf.common.models import ErrorDetails, TelemetryRecord

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import ExportContext
    from aiperf.common.models import (
        MetricResult,
        TelemetryExportData,
    )


@runtime_checkable
class GPUTelemetryCollectorProtocol(Protocol):
    """Protocol for GPU telemetry collectors."""

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
        """Check if the collector source is available."""
        ...

    async def collect_and_process_metrics(self) -> None:
        """Perform a one-shot scrape and dispatch records via the configured callback."""
        ...

    @classmethod
    def validate_environment(cls) -> None:
        """Raise RuntimeError if this collector cannot run on the current host."""
        ...


TRecordCallback = Callable[[list[TelemetryRecord], str], Awaitable[None]]
TErrorCallback = Callable[[ErrorDetails, str], Awaitable[None]]


@runtime_checkable
class GPUTelemetryAccumulatorProtocol(Protocol):
    """Protocol for GPU telemetry accumulators and realtime telemetry."""

    async def process_record(self, record: TelemetryRecord) -> None:
        """Process one GPU telemetry sample."""
        ...

    async def summarize(self) -> list[MetricResult]: ...

    async def export_results(self, ctx: ExportContext) -> TelemetryExportData | None:
        """Export accumulated telemetry data scoped to ``ctx``."""
        ...

    def start_realtime_telemetry(self) -> None:
        """Start realtime telemetry publishing."""
        ...

    def available_platforms(self) -> set[str]:
        """Return the set of GPU platforms (e.g. ``"nvidia"``, ``"amd"``) with data."""
        ...

    def total_power_watts(
        self, start_ns: int | None, end_ns: int | None, platform: str | None = None
    ) -> tuple[float, int]:
        """Per-vendor total of avg(power) over ``[start_ns, end_ns]``."""
        ...

    def total_energy_joules(
        self, start_ns: int | None, end_ns: int | None, platform: str | None = None
    ) -> tuple[float, int]:
        """Per-vendor total energy (J) over ``[start_ns, end_ns]``."""
        ...
