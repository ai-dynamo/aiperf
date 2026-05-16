# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import msgspec

from aiperf.common.enums import MessageType, WorkerStartupState, WorkerStatus
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats


class WorkerHealthMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.WORKER_HEALTH.value
):
    """Worker health check."""

    health: ProcessHealth
    task_stats: WorkerTaskStats

    @property
    def error_rate(self) -> float:
        if self.task_stats.total == 0:
            return 0
        return self.task_stats.failed / self.task_stats.total


class WorkerStatusSummaryMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.WORKER_STATUS_SUMMARY.value
):
    """Aggregate worker status by worker_id."""

    worker_statuses: dict[str, WorkerStatus]
    worker_startup_states: dict[str, WorkerStartupState] = msgspec.field(
        default_factory=dict
    )


class WorkerPodStateMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.WORKER_POD_STATE.value
):
    """Controller-facing aggregate snapshot for a Kubernetes worker pod."""

    pod_index: str
    declared_workers: int
    declared_record_processors: int
    pod_state: str
    admission_state: str
    benchmark_generation: str | None = None
    dataset_generation: str | None = None
    router_connected_workers: int = 0
    dispatchable_workers: int = 0
    ready_workers: int = 0
    ready_record_processors: int = 0
    degraded_workers: int = 0
    degraded_record_processors: int = 0


class WorkerStartupStateMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.WORKER_STARTUP_STATE.value
):
    """Worker startup lifecycle transition.

    Inlines ``request_ns`` with a default_factory to avoid the
    multi-Struct-inheritance pattern (see gotcha_msgspec_multiple_struct_inheritance).
    """

    startup_state: WorkerStartupState
    request_ns: int = msgspec.field(default_factory=time.time_ns)  # type: ignore[assignment]


class WorkerGroupStatsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.WORKER_GROUP_STATS.value
):
    """Aggregate stats for a single worker-group manager.

    Per-worker maps (statuses, startup states, task stats, health) are carried
    inline so the controller can populate ``WorkerGroupStats.workers`` with
    full ``WorkerStats`` for the per-child dropdown rendered by the local
    web UI when exactly one group exists.
    """

    group_id: str
    status: WorkerStatus
    task_stats: WorkerTaskStats
    startup_state: WorkerStartupState | None = None
    declared_workers: int = 0
    ready_workers: int = 0
    health: ProcessHealth | None = None
    worker_statuses: dict[str, WorkerStatus] = msgspec.field(default_factory=dict)
    worker_startup_states: dict[str, WorkerStartupState] = msgspec.field(
        default_factory=dict
    )
    worker_task_stats: dict[str, WorkerTaskStats] = msgspec.field(default_factory=dict)
    worker_health: dict[str, ProcessHealth] = msgspec.field(default_factory=dict)
    last_update_ns: int = msgspec.field(default_factory=time.time_ns)  # type: ignore[assignment]
