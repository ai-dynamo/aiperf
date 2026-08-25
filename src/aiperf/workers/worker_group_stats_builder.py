# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregation helper that rolls up per-worker info into a WorkerGroupStatsMessage.

Lives in its own module to keep ``worker_pod_helpers`` under the ergonomics
file-size cap. Import ``build_worker_group_stats`` from here directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from aiperf.common.enums import WorkerStartupState
from aiperf.common.messages import WorkerGroupStatsMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats
from aiperf.workers.worker_group_state import worst_status

if TYPE_CHECKING:
    from aiperf.workers.worker_group_state import WorkerStatusInfo


def build_worker_group_stats(
    *,
    service_id: str,
    declared_workers: int,
    worker_infos: Mapping[str, WorkerStatusInfo],
) -> WorkerGroupStatsMessage:
    """Aggregate per-child status into a single WorkerGroupStatsMessage.

    - ``task_stats`` summed (total/failed/completed; ``in_progress`` is derived).
    - ``health.cpu_usage`` averaged across children with a non-None health.
    - ``health.memory_usage`` summed.
    - Group status = worst child status (ERROR > HIGH_LOAD > STALE > HEALTHY > IDLE).
    - ``ready_workers`` = count of children with ``startup_state == READY``.
    """
    statuses = {wid: info.status for wid, info in worker_infos.items()}
    startup_states = {
        wid: info.startup_state
        for wid, info in worker_infos.items()
        if info.startup_state is not None
    }
    task_stats_map = {wid: info.task_stats for wid, info in worker_infos.items()}
    health_map = {
        wid: info.health
        for wid, info in worker_infos.items()
        if info.health is not None
    }

    total = sum(t.total for t in task_stats_map.values())
    failed = sum(t.failed for t in task_stats_map.values())
    completed = sum(t.completed for t in task_stats_map.values())
    aggregated_task_stats = WorkerTaskStats(
        total=total, failed=failed, completed=completed
    )

    aggregated_health: ProcessHealth | None = None
    if health_map:
        cpu_avg = sum(h.cpu_usage for h in health_map.values()) / len(health_map)
        mem_sum = sum(h.memory_usage for h in health_map.values())
        first = next(iter(health_map.values()))
        aggregated_health = ProcessHealth(
            pid=first.pid,
            create_time=first.create_time,
            uptime=max(h.uptime for h in health_map.values()),
            cpu_usage=cpu_avg,
            memory_usage=mem_sum,
        )

    group_status = worst_status(info.status for info in worker_infos.values())
    ready_workers = sum(
        1 for s in startup_states.values() if s == WorkerStartupState.READY
    )

    return WorkerGroupStatsMessage(
        service_id=service_id,
        group_id=service_id,
        status=group_status,
        startup_state=None,
        declared_workers=declared_workers,
        ready_workers=ready_workers,
        health=aggregated_health,
        task_stats=aggregated_task_stats,
        worker_statuses=statuses,
        worker_startup_states=startup_states,
        worker_task_stats=task_stats_map,
        worker_health=dict(health_map),
    )
