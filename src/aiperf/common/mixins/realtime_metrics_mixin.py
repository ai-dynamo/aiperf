# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import MessageType
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import RealtimeMetricsMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import MetricResult

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


@provides_hooks(AIPerfHook.ON_REALTIME_METRICS)
class RealtimeMetricsMixin(MessageBusClientMixin):
    """A mixin that provides a hook for real-time metrics."""

    def __init__(self, run: BenchmarkRun, **kwargs):
        super().__init__(run=run, **kwargs)
        self._metrics: list[MetricResult] = []

    @on_message(MessageType.REALTIME_METRICS)
    async def _on_realtime_metrics(self, message: RealtimeMetricsMessage):
        """Update the metrics from a real-time metrics message.

        Lock-free because self._metrics is atomically replaced.
        Operations are atomic only when used in a single thread asyncio context.
        """
        self._metrics = message.metrics
        await self.run_hooks(
            AIPerfHook.ON_REALTIME_METRICS,
            metrics=message.metrics,
        )
