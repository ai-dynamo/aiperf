# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import MessageType, WorkerStartupState, WorkerStatus
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import (
    WorkerGroupStatsMessage,
    WorkerHealthMessage,
    WorkerStatusSummaryMessage,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import (
    ProcessHealth,
    WorkerGroupStats,
    WorkerStats,
    WorkerTaskStats,
)

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


_FAKE_GROUP_ID = "local"
"""Synthetic group id used in fake-in-process mode where no real WGM exists."""


class WorkerGroupTracker:
    """Standalone per-group worker tracker.

    Keyed by ``group_id`` (the WorkerGroupManager service_id). In fake-in-process
    mode (no WGM), per-worker WORKER_HEALTH messages are folded into a single
    synthetic group ``"local"`` so the dashboard still has a row to render.
    """

    def __init__(self) -> None:
        self._groups: dict[str, WorkerGroupStats] = {}

    def update_from_group_message(
        self, message: WorkerGroupStatsMessage
    ) -> WorkerGroupStats:
        """Replace a group entry from a freshly-published WGM snapshot."""
        children: dict[str, WorkerStats] = {}
        for wid, status in message.worker_statuses.items():
            children[wid] = WorkerStats(
                worker_id=wid,
                status=status,
                startup_state=message.worker_startup_states.get(wid),
                health=message.worker_health.get(wid),
                task_stats=message.worker_task_stats.get(wid, WorkerTaskStats()),
                last_update_ns=message.last_update_ns,
            )
        group = WorkerGroupStats(
            group_id=message.group_id,
            status=message.status,
            startup_state=message.startup_state,
            declared_workers=message.declared_workers,
            ready_workers=message.ready_workers,
            health=message.health,
            task_stats=message.task_stats,
            workers=children,
            last_update_ns=message.last_update_ns,
        )
        self._groups[message.group_id] = group
        return group

    def update_from_worker_health(
        self,
        worker_id: str,
        health: ProcessHealth,
        task_stats: WorkerTaskStats,
    ) -> WorkerGroupStats:
        """Fold a per-worker health message into the synthetic ``local`` group.

        Only used by fake-in-process tests where workers publish
        WORKER_HEALTH directly (no WorkerGroupManager exists).
        """
        group = self._groups.get(_FAKE_GROUP_ID) or WorkerGroupStats(
            group_id=_FAKE_GROUP_ID
        )
        children = dict(group.workers)
        children[worker_id] = WorkerStats(
            worker_id=worker_id,
            status=WorkerStatus.HEALTHY,
            health=health,
            task_stats=task_stats,
        )
        group.workers = children
        group.task_stats = WorkerTaskStats(
            total=sum(c.task_stats.total for c in children.values()),
            failed=sum(c.task_stats.failed for c in children.values()),
        )
        if children:
            healthy_children = [c for c in children.values() if c.health is not None]
            if healthy_children:
                first = healthy_children[0].health
                group.health = ProcessHealth(
                    pid=first.pid,
                    create_time=first.create_time,
                    uptime=max(c.health.uptime for c in healthy_children),
                    cpu_usage=sum(c.health.cpu_usage for c in healthy_children)
                    / len(healthy_children),
                    memory_usage=sum(c.health.memory_usage for c in healthy_children),
                )
        group.declared_workers = max(group.declared_workers, len(children))
        self._groups[_FAKE_GROUP_ID] = group
        return group

    def update_worker_statuses(self, worker_statuses: dict[str, WorkerStatus]) -> None:
        """Update per-worker statuses (legacy WORKER_STATUS_SUMMARY path).

        Folded under the synthetic ``local`` group when there is no
        matching WGM-keyed group yet.
        """
        group = self._groups.get(_FAKE_GROUP_ID) or WorkerGroupStats(
            group_id=_FAKE_GROUP_ID
        )
        children = dict(group.workers)
        for worker_id, status in worker_statuses.items():
            child = children.get(worker_id) or WorkerStats(worker_id=worker_id)
            child.status = status
            children[worker_id] = child
        group.workers = children
        self._groups[_FAKE_GROUP_ID] = group

    def update_worker_startup_states(
        self, worker_startup_states: dict[str, WorkerStartupState]
    ) -> None:
        group = self._groups.get(_FAKE_GROUP_ID) or WorkerGroupStats(
            group_id=_FAKE_GROUP_ID
        )
        children = dict(group.workers)
        for worker_id, startup_state in worker_startup_states.items():
            child = children.get(worker_id) or WorkerStats(worker_id=worker_id)
            child.startup_state = startup_state
            children[worker_id] = child
        group.workers = children
        self._groups[_FAKE_GROUP_ID] = group

    def get_group(self, group_id: str) -> WorkerGroupStats | None:
        return self._groups.get(group_id)

    @property
    def worker_groups(self) -> dict[str, WorkerGroupStats]:
        return self._groups


@provides_hooks(
    AIPerfHook.ON_WORKER_GROUP_UPDATE,
    AIPerfHook.ON_WORKER_UPDATE,
    AIPerfHook.ON_WORKER_STATUS_SUMMARY,
)
class WorkerTrackerMixin(MessageBusClientMixin):
    """Tracks worker-group health/stats via the message bus."""

    def __init__(self, run: BenchmarkRun, **kwargs):
        super().__init__(run=run, **kwargs)
        self._worker_tracker = WorkerGroupTracker()

    @on_message(MessageType.WORKER_GROUP_STATS)
    async def _on_worker_group_stats(self, message: WorkerGroupStatsMessage) -> None:
        group = self._worker_tracker.update_from_group_message(message)
        await self.run_hooks(
            AIPerfHook.ON_WORKER_GROUP_UPDATE,
            group_id=group.group_id,
            group_stats=group,
        )

    @on_message(MessageType.WORKER_HEALTH)
    async def _on_worker_health(self, message: WorkerHealthMessage) -> None:
        """Fake-in-process fallback: worker publishes raw health, no WGM."""
        group = self._worker_tracker.update_from_worker_health(
            message.service_id, message.health, message.task_stats
        )
        await self.run_hooks(
            AIPerfHook.ON_WORKER_GROUP_UPDATE,
            group_id=group.group_id,
            group_stats=group,
        )
        # Keep legacy hook firing so any external listener still works.
        await self.run_hooks(
            AIPerfHook.ON_WORKER_UPDATE,
            worker_id=message.service_id,
            worker_stats=group.workers[message.service_id],
        )

    @on_message(MessageType.WORKER_STATUS_SUMMARY)
    async def _on_worker_status_summary(
        self, message: WorkerStatusSummaryMessage
    ) -> None:
        self._worker_tracker.update_worker_statuses(message.worker_statuses)
        self._worker_tracker.update_worker_startup_states(message.worker_startup_states)
        await self.run_hooks(
            AIPerfHook.ON_WORKER_STATUS_SUMMARY,
            worker_status_summary=message.worker_statuses,
            worker_startup_states=message.worker_startup_states,
        )
