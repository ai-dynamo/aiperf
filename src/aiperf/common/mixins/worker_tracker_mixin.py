# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import MessageType, WorkerStatus
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import WorkerHealthMessage, WorkerStatusSummaryMessage
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import ProcessHealth, WorkerStats, WorkerTaskStats


class WorkerTracker:
    """Standalone worker tracker for tracking worker health and stats."""

    def __init__(self) -> None:
        self._workers_stats: dict[str, WorkerStats] = {}

    def update_worker_stats(
        self, worker_id: str, health: ProcessHealth, task_stats: WorkerTaskStats
    ) -> WorkerStats:
        """Update worker health and task stats, returns the updated WorkerStats."""
        if worker_id not in self._workers_stats:
            self._workers_stats[worker_id] = WorkerStats(worker_id=worker_id)
        self._workers_stats[worker_id].health = health
        self._workers_stats[worker_id].task_stats = task_stats
        return self._workers_stats[worker_id]

    def update_worker_statuses(self, worker_statuses: dict[str, WorkerStatus]) -> None:
        """Update worker statuses from a status summary."""
        for worker_id, status in worker_statuses.items():
            if worker_id not in self._workers_stats:
                self._workers_stats[worker_id] = WorkerStats(worker_id=worker_id)
            self._workers_stats[worker_id].status = status

    def get_worker_stats(self, worker_id: str) -> WorkerStats | None:
        """Get stats for a specific worker."""
        return self._workers_stats.get(worker_id)

    @property
    def workers(self) -> dict[str, WorkerStats]:
        """All tracked workers."""
        return self._workers_stats


@provides_hooks(AIPerfHook.ON_WORKER_UPDATE, AIPerfHook.ON_WORKER_STATUS_SUMMARY)
class WorkerTrackerMixin(MessageBusClientMixin):
    """A worker tracker mixin that tracks the health and tasks of workers via message bus."""

    def __init__(self, service_config: ServiceConfig, **kwargs):
        super().__init__(service_config=service_config, **kwargs)
        self._worker_tracker = WorkerTracker()

    @on_message(MessageType.WORKER_HEALTH)
    async def _on_worker_health(self, message: WorkerHealthMessage):
        """Update the worker stats from a worker health message."""
        worker_stats = self._worker_tracker.update_worker_stats(
            message.service_id, message.health, message.task_stats
        )
        await self.run_hooks(
            AIPerfHook.ON_WORKER_UPDATE,
            worker_id=message.service_id,
            worker_stats=worker_stats,
        )

    @on_message(MessageType.WORKER_STATUS_SUMMARY)
    async def _on_worker_status_summary(self, message: WorkerStatusSummaryMessage):
        """Update the worker stats from a worker status summary message."""
        self._worker_tracker.update_worker_statuses(message.worker_statuses)
        await self.run_hooks(
            AIPerfHook.ON_WORKER_STATUS_SUMMARY,
            worker_status_summary=message.worker_statuses,
        )
