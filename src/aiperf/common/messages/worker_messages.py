# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from pydantic import Field

from aiperf.common.enums import MessageType, WorkerStartupState, WorkerStatus
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ProcessHealth, WorkerTaskStats
from aiperf.common.types import MessageTypeT


class WorkerHealthMessage(BaseServiceMessage):
    """Message for a worker health check."""

    message_type: MessageTypeT = MessageType.WORKER_HEALTH

    health: ProcessHealth = Field(..., description="The health of the worker process")

    # Worker specific fields
    task_stats: WorkerTaskStats = Field(
        ...,
        description="Stats for the tasks that have been sent to the worker",
    )

    @property
    def error_rate(self) -> float:
        """The error rate of the worker."""
        if self.task_stats.total == 0:
            return 0
        return self.task_stats.failed / self.task_stats.total


class WorkerStatusSummaryMessage(BaseServiceMessage):
    """Message for a worker status summary."""

    message_type: MessageTypeT = MessageType.WORKER_STATUS_SUMMARY

    worker_statuses: dict[str, WorkerStatus] = Field(
        ...,
        description="A mapping of worker IDs to their status",
    )
    worker_startup_states: dict[str, WorkerStartupState] = Field(
        default_factory=dict,
        description="A mapping of worker IDs to their startup state",
    )


class WorkerPodStateMessage(BaseServiceMessage):
    """Controller-facing aggregate snapshot for a Kubernetes worker pod."""

    message_type: MessageTypeT = MessageType.WORKER_POD_STATE

    pod_index: str = Field(..., description="Pod index (e.g. ordinal in StatefulSet)")
    declared_workers: int = Field(
        ..., ge=0, description="Workers declared in the pod's spec"
    )
    declared_record_processors: int = Field(
        ..., ge=0, description="Record processors declared in the pod's spec"
    )
    pod_state: str = Field(..., description="Coarse pod lifecycle state")
    admission_state: str = Field(
        ..., description="WGM admission state (e.g. probing/ready/full)"
    )
    benchmark_generation: str | None = Field(
        None, description="Benchmark generation tag this pod last reported on"
    )
    dataset_generation: str | None = Field(
        None, description="Dataset generation tag this pod last reported on"
    )
    router_connected_workers: int = Field(
        0, ge=0, description="Workers that have completed router probe"
    )
    dispatchable_workers: int = Field(
        0, ge=0, description="Workers eligible to receive credits"
    )
    ready_workers: int = Field(
        0, ge=0, description="Workers in READY startup state"
    )
    ready_record_processors: int = Field(
        0, ge=0, description="Record processors in READY state"
    )
    degraded_workers: int = Field(
        0, ge=0, description="Workers in degraded state"
    )
    degraded_record_processors: int = Field(
        0, ge=0, description="Record processors in degraded state"
    )


class WorkerStartupStateMessage(BaseServiceMessage):
    """Worker startup lifecycle transition."""

    message_type: MessageTypeT = MessageType.WORKER_STARTUP_STATE

    startup_state: WorkerStartupState = Field(
        ..., description="The worker's current startup state"
    )
    request_ns: int = Field(
        default_factory=time.time_ns,
        ge=0,
        description="Nanosecond timestamp when this transition was reported",
    )


class WorkerGroupStatsMessage(BaseServiceMessage):
    """Aggregate stats for a single worker-group manager.

    Per-worker maps (statuses, startup states, task stats, health) are carried
    inline so the controller can populate full per-worker stats for the
    per-child dropdown rendered by the local web UI when exactly one group
    exists.
    """

    message_type: MessageTypeT = MessageType.WORKER_GROUP_STATS

    group_id: str = Field(..., description="Worker group manager identifier")
    status: WorkerStatus = Field(..., description="Aggregated group status")
    task_stats: WorkerTaskStats = Field(
        ..., description="Aggregated task stats across the group"
    )
    startup_state: WorkerStartupState | None = Field(
        None, description="Aggregated startup state for the group, if known"
    )
    declared_workers: int = Field(
        0, ge=0, description="Workers declared in the group"
    )
    ready_workers: int = Field(
        0, ge=0, description="Workers in READY startup state"
    )
    health: ProcessHealth | None = Field(
        None, description="Aggregated group health, if known"
    )
    worker_statuses: dict[str, WorkerStatus] = Field(
        default_factory=dict, description="Per-worker status map"
    )
    worker_startup_states: dict[str, WorkerStartupState] = Field(
        default_factory=dict, description="Per-worker startup-state map"
    )
    worker_task_stats: dict[str, WorkerTaskStats] = Field(
        default_factory=dict, description="Per-worker task stats map"
    )
    worker_health: dict[str, ProcessHealth] = Field(
        default_factory=dict, description="Per-worker health map"
    )
    last_update_ns: int = Field(
        default_factory=time.time_ns,
        ge=0,
        description="Nanosecond timestamp when these stats were sampled",
    )
