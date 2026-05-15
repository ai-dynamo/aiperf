# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server metrics router: real-time Prometheus metrics from inference servers."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter

from aiperf.api.routers.base_router import BaseRouter, component_dependency
from aiperf.common.enums import MessageType
from aiperf.common.hooks import on_message
from aiperf.common.messages import RealtimeServerMetricsMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin

ServerMetricsDep = Annotated[
    "ServerMetricsRouter", component_dependency("server_metrics")
]

server_metrics_router = APIRouter()


class ServerMetricsRouter(MessageBusClientMixin, BaseRouter):
    """Receives real-time server metrics from the message bus and serves them via REST."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest: dict[str, Any] | None = None

    def get_router(self) -> APIRouter:
        return server_metrics_router

    @on_message(MessageType.REALTIME_SERVER_METRICS)
    async def _on_realtime_server_metrics(
        self, message: RealtimeServerMetricsMessage
    ) -> None:
        self._latest = message.model_dump(
            mode="json", exclude={"message_type", "service_id"}
        )


@server_metrics_router.get("/api/server-metrics", tags=["Server Metrics"])
async def get_server_metrics(component: ServerMetricsDep) -> dict[str, Any]:
    """Get real-time server metrics from Prometheus endpoints.

    Returns server-side metrics (queue depth, cache usage, latency, throughput)
    collected from the inference server's Prometheus endpoint during the benchmark.
    """
    if component._latest is None:
        return {"endpoint_summaries": {}, "message": "No server metrics available yet"}
    return component._latest
