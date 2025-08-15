# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import MessageType
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import RealtimeMetricsMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import MetricResult
from aiperf.controller.system_controller import SystemController


@provides_hooks(AIPerfHook.ON_REALTIME_METRICS)
class RealtimeMetricsMixin(MessageBusClientMixin):
    """A mixin that provides a hook for real-time metrics."""

    def __init__(
        self, service_config: ServiceConfig, controller: SystemController, **kwargs
    ):
        super().__init__(service_config=service_config, controller=controller, **kwargs)
        self._controller = controller
        self._metrics: list[MetricResult] = []
        self._metrics_lock = asyncio.Lock()

    @on_message(MessageType.REALTIME_METRICS)
    async def _on_realtime_metrics(self, message: RealtimeMetricsMessage):
        """Update the metrics from a real-time metrics message."""
        async with self._metrics_lock:
            self._metrics = message.metrics
        await self.run_hooks(
            AIPerfHook.ON_REALTIME_METRICS,
            metrics=message.metrics,
        )
