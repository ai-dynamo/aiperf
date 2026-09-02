# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import ExportContext
    from aiperf.common.models import (
        MetricResult,
        ServerMetricsEndpointSummary,
        ServerMetricsRecord,
        ServerMetricsResults,
    )


@runtime_checkable
class ServerMetricsAccumulatorProtocol(Protocol):
    """Protocol for server metrics accumulators and realtime exporters."""

    async def process_record(self, record: ServerMetricsRecord) -> None:
        """Process one Prometheus server-metrics snapshot."""
        ...

    async def summarize(self) -> list[MetricResult]: ...

    async def export_results(self, ctx: ExportContext) -> ServerMetricsResults | None:
        """Export accumulated server metrics scoped to ``ctx``."""
        ...

    def compute_endpoint_summaries(
        self,
        profiling_start_ns: int,
        profiling_end_ns: int,
        slice_duration: float | None = None,
        *,
        include_final_collection: bool = True,
    ) -> dict[str, ServerMetricsEndpointSummary]:
        """Compute endpoint summaries for a bounded realtime message."""
        ...

    def realtime_snapshot(self, start_ns: int | None = None) -> dict[str, float]:
        """Return the compact scalar fields used by realtime consumers."""
        ...
