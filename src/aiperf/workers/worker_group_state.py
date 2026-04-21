# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared worker/group status state helpers."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Protocol

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import WorkerStartupState, WorkerStatus
from aiperf.common.environment import Environment
from aiperf.common.messages import WorkerHealthMessage
from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage
from aiperf.common.models.progress_models import WorkerStats


class WorkerStatusSnapshot(Protocol):
    """Protocol shared by worker/group status summary builders."""

    status: WorkerStatus
    startup_state: WorkerStartupState | None


class WorkerStatusInfo(WorkerStats, kw_only=True):
    """Shared worker-status state used by WorkerManager compatibility paths."""

    last_error_ns: int | None = None
    last_high_load_ns: int | None = None


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


__all__ = [
    "WorkerStatusInfo",
    "build_worker_status_summary",
    "mark_stale_workers",
    "update_worker_status",
]
