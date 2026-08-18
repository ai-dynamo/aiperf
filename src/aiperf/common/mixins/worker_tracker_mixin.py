# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from aiperf.common.enums import MessageType, WorkerStatus
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

LOCAL_GROUP_ID = "local"
"""Synthetic group id used when no WorkerGroupManager exists.

Exposed (rather than module-private) so API routers and dashboard renderers can
reference the same constant when distinguishing the synthetic in-process group
from real WGM-keyed groups.
"""

STATUS_RANK: dict[WorkerStatus, int] = {
    WorkerStatus.IDLE: 0,
    WorkerStatus.HEALTHY: 1,
    WorkerStatus.HIGH_LOAD: 2,
    WorkerStatus.STALE: 3,
    WorkerStatus.ERROR: 4,
}
"""Precedence used to roll child statuses up into a group status.

Higher rank wins. ``STALE`` outranks ``HIGH_LOAD`` because a missing heartbeat
means the worker's real state is unknown, which is worse than a known-saturated
but still-reporting worker.
"""


def worst_status(statuses: Iterable[WorkerStatus]) -> WorkerStatus:
    """Return the highest-precedence status from ``statuses`` per ``STATUS_RANK``.

    Empty input returns ``WorkerStatus.IDLE`` (no workers, no concern).

    Example:
        >>> worst_status([WorkerStatus.HEALTHY, WorkerStatus.ERROR])
        <WorkerStatus.ERROR: 'error'>
    """
    materialized = list(statuses)
    if not materialized:
        return WorkerStatus.IDLE
    return max(materialized, key=lambda s: STATUS_RANK.get(s, 0))


class WorkerTracker:
    """Tracks worker health and stats, keyed by worker group.

    Groups are keyed by ``group_id`` (the WorkerGroupManager service id). When
    workers publish ``WORKER_HEALTH``/``WORKER_STATUS_SUMMARY`` directly and no
    WorkerGroupManager exists (in-process mode), their stats are folded into a
    single synthetic ``"local"`` group so consumers always have a group to
    render. The flat :attr:`workers` view remains available for callers that do
    not care about grouping.
    """

    def __init__(self) -> None:
        self._groups: dict[str, WorkerGroupStats] = {}

    def _local_group(self) -> WorkerGroupStats:
        """Return the synthetic local group, creating it if absent."""
        group = self._groups.get(LOCAL_GROUP_ID)
        if group is None:
            group = WorkerGroupStats(group_id=LOCAL_GROUP_ID)
            self._groups[LOCAL_GROUP_ID] = group
        return group

    @property
    def _workers_stats(self) -> dict[str, WorkerStats]:
        """Expose the legacy local-worker storage seam during API migration."""
        return self._local_group().workers

    @_workers_stats.setter
    def _workers_stats(self, workers: dict[str, WorkerStats]) -> None:
        group = self._local_group()
        group.workers = workers
        self._refresh_local_rollup(group)

    def update_from_group_message(
        self, message: WorkerGroupStatsMessage
    ) -> WorkerGroupStats:
        """Replace a group entry from a freshly-published WGM snapshot."""
        children = {
            worker_id: WorkerStats(
                worker_id=worker_id,
                status=status,
                startup_state=message.worker_startup_states.get(worker_id),
                health=message.worker_health.get(worker_id),
                task_stats=message.worker_task_stats.get(worker_id, WorkerTaskStats()),
                last_update_ns=message.last_update_ns,
            )
            for worker_id, status in message.worker_statuses.items()
        }
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

    def update_worker_stats(
        self, worker_id: str, health: ProcessHealth, task_stats: WorkerTaskStats
    ) -> WorkerStats:
        """Update worker health and task stats, returns the updated WorkerStats."""
        group = self._local_group()
        worker = group.workers.get(worker_id)
        if worker is None:
            worker = WorkerStats(worker_id=worker_id)
            group.workers[worker_id] = worker
        worker.health = health
        worker.task_stats = task_stats
        self._refresh_local_rollup(group)
        return worker

    def update_worker_statuses(self, worker_statuses: dict[str, WorkerStatus]) -> None:
        """Update worker statuses from a status summary."""
        group = self._local_group()
        for worker_id, status in worker_statuses.items():
            worker = group.workers.get(worker_id)
            if worker is None:
                worker = WorkerStats(worker_id=worker_id)
                group.workers[worker_id] = worker
            worker.status = status
        self._refresh_local_rollup(group)

    def _refresh_local_rollup(self, group: WorkerGroupStats) -> None:
        """Recompute the synthetic group's aggregate view from its children."""
        children = group.workers.values()
        group.task_stats = WorkerTaskStats(
            total=sum(c.task_stats.total for c in children),
            completed=sum(c.task_stats.completed for c in children),
            failed=sum(c.task_stats.failed for c in children),
        )
        group.declared_workers = max(group.declared_workers, len(group.workers))
        group.status = worst_status(c.status for c in children)

    def get_worker_stats(self, worker_id: str) -> WorkerStats | None:
        """Get stats for a specific worker, searched across every group.

        A worker id is expected to be unique cluster-wide. If the same id
        somehow appears in two groups, the first group in insertion order wins
        -- the same rule :attr:`workers` applies, so the two accessors never
        disagree about which entry is authoritative.
        """
        for group in self._groups.values():
            worker = group.workers.get(worker_id)
            if worker is not None:
                return worker
        return None

    def get_group(self, group_id: str) -> WorkerGroupStats | None:
        """Get aggregate stats for a specific worker group."""
        return self._groups.get(group_id)

    @property
    def worker_groups(self) -> dict[str, WorkerGroupStats]:
        """All tracked worker groups, keyed by group id."""
        return self._groups

    @property
    def workers(self) -> dict[str, WorkerStats]:
        """All tracked workers, flattened across every group.

        Returns a newly-built snapshot dict, not the tracker's live storage:
        mutating the returned mapping (``tracker.workers[id] = ...``) does not
        affect tracker state. The ``WorkerStats`` values are the live objects,
        so mutating a value does. Write through
        :meth:`update_from_group_message` / :meth:`update_worker_stats` instead.

        On a duplicate worker id across groups the first group in insertion
        order wins, matching :meth:`get_worker_stats`.
        """
        flattened: dict[str, WorkerStats] = {}
        for group in self._groups.values():
            for worker_id, worker in group.workers.items():
                flattened.setdefault(worker_id, worker)
        return flattened


@provides_hooks(
    AIPerfHook.ON_WORKER_GROUP_UPDATE,
    AIPerfHook.ON_WORKER_UPDATE,
    AIPerfHook.ON_WORKER_STATUS_SUMMARY,
)
class WorkerTrackerMixin(MessageBusClientMixin):
    """A worker tracker mixin that tracks the health and tasks of workers via message bus."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._worker_tracker = WorkerTracker()

    @on_message(MessageType.WORKER_GROUP_STATS)
    async def _on_worker_group_stats(self, message: WorkerGroupStatsMessage) -> None:
        """Replace a group's stats from a WorkerGroupManager snapshot."""
        group = self._worker_tracker.update_from_group_message(message)
        await self.run_hooks(
            AIPerfHook.ON_WORKER_GROUP_UPDATE,
            group_id=group.group_id,
            group_stats=group,
        )

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
