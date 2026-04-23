# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result publication helper for ServerMetricsManager."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

from aiperf.common.messages import ProcessServerMetricsResultMessage
from aiperf.common.models import ErrorDetailsCount, ProcessServerMetricsResult

if TYPE_CHECKING:
    from aiperf.common.models import ErrorTrackingState
    from aiperf.server_metrics.protocols import ServerMetricsAccumulatorProtocol


class _Publisher(Protocol):
    service_id: str

    async def publish(self, msg: ProcessServerMetricsResultMessage) -> None: ...
    def warning(self, msg: str) -> None: ...


async def publish_server_metrics_result(
    *,
    publisher: _Publisher,
    accumulator: ServerMetricsAccumulatorProtocol | None,
    error_state: ErrorTrackingState,
    start_ns: int | None,
    end_ns: int | None,
) -> None:
    """Publish accumulated server metrics results. Caller handles idempotency."""
    error_summary = [
        ErrorDetailsCount(error_details=err, count=count)
        for err, count in error_state.error_counts.items()
    ]
    if not accumulator:
        await publisher.publish(
            ProcessServerMetricsResultMessage(
                service_id=publisher.service_id,
                server_metrics_result=ProcessServerMetricsResult(results=None),
            )
        )
        return

    resolved_start_ns = start_ns or time.time_ns()
    resolved_end_ns = end_ns or time.time_ns()
    if resolved_end_ns < resolved_start_ns:
        publisher.warning(
            f"Invalid time window start_ns={resolved_start_ns} > end_ns={resolved_end_ns}; "
            "falling back to full-history export (start_ns=0)"
        )
        resolved_start_ns = 0

    export_data = await accumulator.export_results(
        start_ns=resolved_start_ns,
        end_ns=resolved_end_ns,
        error_summary=error_summary,
    )
    await publisher.publish(
        ProcessServerMetricsResultMessage(
            service_id=publisher.service_id,
            server_metrics_result=ProcessServerMetricsResult(results=export_data),
        )
    )
