# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import MessageType, WorkerStartupState, WorkerStatus
from aiperf.common.messages.base_messages import RequiresRequestNSMixin
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
        description="A mapping of worker IDs to their startup lifecycle state",
    )


class WorkerPodStateMessage(BaseServiceMessage):
    """Controller-facing aggregate snapshot for a Kubernetes worker pod."""

    message_type: MessageTypeT = MessageType.WORKER_POD_STATE

    pod_index: str = Field(..., description="The Kubernetes worker pod index.")
    benchmark_generation: str | None = Field(
        default=None,
        description="The current benchmark generation loaded by this pod.",
    )
    dataset_generation: str | None = Field(
        default=None,
        description="The current dataset generation loaded by this pod.",
    )
    declared_workers: int = Field(
        ..., description="Configured worker count declared by this pod."
    )
    declared_record_processors: int = Field(
        ..., description="Configured record-processor count declared by this pod."
    )
    router_connected_workers: int = Field(
        default=0,
        description="Workers that have connected to the credit router.",
    )
    dispatchable_workers: int = Field(
        default=0,
        description="Workers currently eligible to receive credits.",
    )
    ready_workers: int = Field(
        default=0,
        description="Workers that have fully completed startup.",
    )
    ready_record_processors: int = Field(
        default=0,
        description="Record processors currently available in the pod.",
    )
    degraded_workers: int = Field(
        default=0,
        description="Configured workers that are not currently ready.",
    )
    degraded_record_processors: int = Field(
        default=0,
        description="Configured record processors that are not currently ready.",
    )
    pod_state: str = Field(..., description="Aggregate worker-pod lifecycle state.")
    admission_state: str = Field(
        ..., description="Whether this pod is admitting new work."
    )


class WorkerStartupStateMessage(BaseServiceMessage, RequiresRequestNSMixin):
    """Message for a worker startup lifecycle transition."""

    message_type: MessageTypeT = MessageType.WORKER_STARTUP_STATE

    startup_state: WorkerStartupState = Field(
        ...,
        description="The current startup lifecycle state of the worker.",
    )
