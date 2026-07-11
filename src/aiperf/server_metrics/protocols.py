# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aiperf.common.models import (
        ErrorDetailsCount,
        MetricResult,
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

    async def export_results(
        self,
        start_ns: int,
        end_ns: int,
        error_summary: list[ErrorDetailsCount] | None = None,
        *,
        warmup_start_ns: int | None = None,
        warmup_end_ns: int | None = None,
    ) -> ServerMetricsResults | None:
        """Export accumulated server metrics for a profiling window."""
        ...
