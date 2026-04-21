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
