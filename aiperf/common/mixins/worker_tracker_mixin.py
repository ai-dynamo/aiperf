# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import MessageType
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import WorkerHealthMessage, WorkerStatusSummaryMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import WorkerStats


@provides_hooks(AIPerfHook.ON_WORKER_UPDATE, AIPerfHook.ON_WORKER_STATUS_SUMMARY)
class WorkerTrackerMixin(MessageBusClientMixin):
    """A worker tracker that tracks the health and tasks of the workers."""

    def __init__(self, service_config: ServiceConfig, **kwargs):
        super().__init__(service_config=service_config, **kwargs)
        self._workers_stats: dict[str, WorkerStats] = {}
        self._workers_stats_lock = asyncio.Lock()

    @on_message(MessageType.WORKER_HEALTH)
    async def _on_worker_health(self, message: WorkerHealthMessage):
        """Update the worker stats from a worker health message."""
        worker_id = message.service_id
        async with self._workers_stats_lock:
            if worker_id not in self._workers_stats:
                self._workers_stats[worker_id] = WorkerStats(worker_id=worker_id)
            self._workers_stats[worker_id].health = message.health
            self._workers_stats[worker_id].task_stats = message.task_stats
            await self.run_hooks(
                AIPerfHook.ON_WORKER_UPDATE,
                worker_id=worker_id,
                worker_stats=self._workers_stats[worker_id],
            )

    @on_message(MessageType.WORKER_STATUS_SUMMARY)
    async def _on_worker_status_summary(self, message: WorkerStatusSummaryMessage):
        """Update the worker stats from a worker status summary message."""
        async with self._workers_stats_lock:
            for worker_id, status in message.worker_statuses.items():
                if worker_id not in self._workers_stats:
                    self.warning(f"Worker {worker_id} not found in worker stats")
                    continue
                self._workers_stats[worker_id].status = status
        await self.run_hooks(
            AIPerfHook.ON_WORKER_STATUS_SUMMARY,
            worker_status_summary=message.worker_statuses,
        )
