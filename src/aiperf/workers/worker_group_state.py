# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared worker/group status state helpers."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.messages import WorkerHealthMessage
from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage
from aiperf.common.models import (
    ProcessHealth,
    ProcessHealthAggregates,
    WorkerTaskStats,
)
from aiperf.common.models.progress_models import WorkerStats


class WorkerStatusSnapshot(Protocol):
    """Protocol shared by worker/group status summary builders."""

    status: WorkerStatus
    startup_state: WorkerStartupState | None


@dataclass(slots=True, kw_only=True)
class WorkerStatusInfo:
    """Mutable per-worker status accumulator owned by a group manager.

    Field-compatible with :class:`~aiperf.common.models.progress_models.WorkerStats`
    (the wire/UI projection) but deliberately a separate slotted dataclass: this
    is rewritten on every health tick, and it carries two derivation-only
    timestamps plus rolling ``health_aggregates`` that the wire type does not.

    Example:
        >>> info = WorkerStatusInfo(worker_id="worker_0")
        >>> info.status
        <WorkerStatus.IDLE: 'idle'>
    """

    worker_id: str
    """Stable identifier of the worker this snapshot describes."""

    task_stats: WorkerTaskStats = field(default_factory=WorkerTaskStats)
    """Latest task counters (total/completed/failed) reported by the worker."""

    health: ProcessHealth | None = None
    """Latest process-health snapshot, or None before the first health message."""

    health_aggregates: ProcessHealthAggregates = field(
        default_factory=ProcessHealthAggregates
    )
    """Rolling min/max/avg aggregates accumulated across health messages."""

    status: WorkerStatus = WorkerStatus.IDLE
    """Status derived from the latest health message by ``update_worker_status``."""

    startup_state: WorkerStartupState | None = None
    """Latest startup state, or None if the worker has not reported one."""

    startup_state_updated_ns: int | None = None
    """Time the startup state last changed, in nanoseconds."""

    last_update_ns: int | None = None
    """Time health was last updated, in nanoseconds."""

    last_error_ns: int | None = None
    """Time the worker last entered ERROR status, in nanoseconds."""

    last_high_load_ns: int | None = None
    """Time the worker last entered HIGH_LOAD status, in nanoseconds."""

    def to_worker_stats(self) -> WorkerStats:
        """Project this accumulator onto the wire/UI ``WorkerStats`` model."""
        return WorkerStats(
            worker_id=self.worker_id,
            task_stats=self.task_stats,
            health=self.health,
            status=self.status,
            startup_state=self.startup_state,
            startup_state_updated_ns=self.startup_state_updated_ns,
            last_update_ns=self.last_update_ns,
        )


def build_worker_status_summary(
    service_id: str,
    worker_infos: Mapping[str, WorkerStatusSnapshot],
) -> WorkerStatusSummaryMessage:
    """Build a worker/startup summary from any status-tracking mapping."""
    return WorkerStatusSummaryMessage(
        service_id=service_id,
        worker_statuses={
            worker_id: info.status for worker_id, info in worker_infos.items()
        },
        worker_startup_states={
            worker_id: info.startup_state
            for worker_id, info in worker_infos.items()
            if info.startup_state is not None
        },
    )


def update_worker_status(
    info: WorkerStatusInfo,
    message: WorkerHealthMessage,
    warning: Callable[[str], None] | None = None,
) -> None:
    """Update a worker status snapshot using the shared status rules."""
    info.last_update_ns = time.time_ns()
    if message.task_stats.failed > info.task_stats.failed:
        info.last_error_ns = time.time_ns()
        info.status = WorkerStatus.ERROR
    elif (time.time_ns() - (info.last_error_ns or 0)) / NANOS_PER_SECOND < Environment.WORKER.ERROR_RECOVERY_TIME:  # fmt: skip
        info.status = WorkerStatus.ERROR
    elif message.health.cpu_usage > Environment.WORKER.HIGH_LOAD_CPU_USAGE:
        info.last_high_load_ns = time.time_ns()
        if warning is not None:
            warning(
                f"CPU usage for {message.service_id} is {round(message.health.cpu_usage)}%. "
                "AIPerf results may be inaccurate."
            )
        info.status = WorkerStatus.HIGH_LOAD
    elif (time.time_ns() - (info.last_high_load_ns or 0)) / NANOS_PER_SECOND < Environment.WORKER.HIGH_LOAD_RECOVERY_TIME:  # fmt: skip
        info.status = WorkerStatus.HIGH_LOAD
    elif message.task_stats.total == 0 or message.task_stats.in_progress == 0:
        info.status = WorkerStatus.IDLE
    else:
        info.status = WorkerStatus.HEALTHY

    info.health = message.health
    info.task_stats = message.task_stats

    aggregates = info.health_aggregates
    aggregates.memory_usage.update(message.health.memory_usage)
    aggregates.cpu_usage.update(message.health.cpu_usage)
    aggregates.num_threads.update(message.health.num_threads)
    if message.health.num_ctx_switches:
        aggregates.voluntary_ctx_switches.update(message.health.num_ctx_switches[0])
        aggregates.involuntary_ctx_switches.update(message.health.num_ctx_switches[1])
    if message.health.io_counters:
        aggregates.io_read_bytes.update(message.health.io_counters[4])
        aggregates.io_write_bytes.update(message.health.io_counters[5])
    if message.health.cpu_times:
        aggregates.cpu_time_user.update(message.health.cpu_times[0])
        aggregates.cpu_time_system.update(message.health.cpu_times[1])
        aggregates.cpu_time_iowait.update(message.health.cpu_times[2])


def mark_stale_workers(worker_infos: Mapping[str, WorkerStatusInfo]) -> None:
    """Mark workers stale when their latest health/startup activity has expired."""
    for info in worker_infos.values():
        last_activity_ns = max(
            info.last_update_ns or 0,
            info.startup_state_updated_ns or 0,
        )
        if last_activity_ns == 0:
            continue
        if (time.time_ns() - last_activity_ns) / NANOS_PER_SECOND > Environment.WORKER.STALE_TIME:  # fmt: skip
            info.status = WorkerStatus.STALE


STATUS_RANK: dict[WorkerStatus, int] = {
    WorkerStatus.IDLE: 0,
    WorkerStatus.HEALTHY: 1,
    WorkerStatus.HIGH_LOAD: 2,
    WorkerStatus.STALE: 3,
    WorkerStatus.ERROR: 4,
}
"""Precedence used to roll up child statuses into a group status.

Higher rank wins when aggregating across workers:

- ``IDLE = 0`` -- no work, no concern.
- ``HEALTHY = 1`` -- actively working, no concern.
- ``HIGH_LOAD = 2`` -- actively working but CPU-saturated; results may be inaccurate.
- ``STALE = 3`` -- no recent heartbeat; we don't know what state it's in, so we
  treat that uncertainty as worse than known HIGH_LOAD.
- ``ERROR = 4`` -- terminal failure observed.
"""


def worst_status(statuses: Iterable[WorkerStatus]) -> WorkerStatus:
    """Return the highest-precedence status from ``statuses`` per ``STATUS_RANK``.

    Empty input returns ``WorkerStatus.IDLE`` (no workers, no concern).
    """
    materialized = list(statuses)
    if not materialized:
        return WorkerStatus.IDLE
    return max(materialized, key=lambda s: STATUS_RANK.get(s, 0))


__all__ = [
    "STATUS_RANK",
    "WorkerStatusInfo",
    "build_worker_status_summary",
    "mark_stale_workers",
    "update_worker_status",
    "worst_status",
]
