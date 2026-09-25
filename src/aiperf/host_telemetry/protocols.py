# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol for host-level power and energy telemetry collectors.

Deliberately the same surface as `GPUTelemetryCollectorProtocol`. Nothing in
that protocol is actually accelerator-specific; it is initialize, start, stop,
collect, and a reachability check, which describes a RAPL reader or a board PMIC
reader as well as it describes DCGM. Only the name and the record type are
GPU-bound.

This is a sibling rather than a widening of the existing category, so that it
adds a host path without changing anything the three shipped GPU collectors
depend on. If the maintainers would rather have one domain-tagged category, the
collectors written against this protocol move across unchanged and only the
registration differs.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from aiperf.common.models import ErrorDetails
from aiperf.common.models.host_telemetry_models import HostTelemetryRecord


@runtime_checkable
class HostTelemetryCollectorProtocol(Protocol):
    """Protocol for host power and energy telemetry collectors."""

    @property
    def id(self) -> str:
        """Get the collector's unique identifier."""
        ...

    @property
    def endpoint_url(self) -> str:
        """Get the source identifier (e.g. 'rapl://localhost')."""
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


THostRecordCallback = Callable[[list[HostTelemetryRecord], str], Awaitable[None]]
THostErrorCallback = Callable[[ErrorDetails, str], Awaitable[None]]
