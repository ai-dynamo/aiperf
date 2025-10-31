# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import MessageType
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import RealtimeTelemetryMetricsMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import MetricResult
from aiperf.controller.system_controller import SystemController


@provides_hooks(AIPerfHook.ON_REALTIME_TELEMETRY_METRICS)
class RealtimeTelemetryMetricsMixin(MessageBusClientMixin):
    """A mixin that provides a hook for real-time GPU telemetry metrics."""

    def __init__(
        self, service_config: ServiceConfig, controller: SystemController, **kwargs
    ):
        super().__init__(service_config=service_config, controller=controller, **kwargs)
        self._controller = controller
        self._telemetry_metrics: list[MetricResult] = []
        self._telemetry_metrics_lock = asyncio.Lock()

    @on_message(MessageType.REALTIME_TELEMETRY_METRICS)
    async def _on_realtime_telemetry_metrics(
        self, message: RealtimeTelemetryMetricsMessage
    ):
        """Update the telemetry metrics from a real-time telemetry metrics message."""
        self.debug(
            f"Mixin received telemetry message with {len(message.metrics)} metrics, triggering hook"
        )

        async with self._telemetry_metrics_lock:
            self._telemetry_metrics = message.metrics
        await self.run_hooks(
            AIPerfHook.ON_REALTIME_TELEMETRY_METRICS,
            metrics=message.metrics,
        )
