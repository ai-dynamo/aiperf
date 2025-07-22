# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import (
    Field,
)

from aiperf.common.enums import (
    CreditPhase,
    MessageType,
)
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ProcessHealth, WorkerPhaseTaskStats
from aiperf.common.types import MessageTypeT


class WorkerHealthMessage(BaseServiceMessage):
    """Message for a worker health check."""

    message_type: MessageTypeT = MessageType.WORKER_HEALTH

    process: ProcessHealth = Field(..., description="The health of the worker process")

    # Worker specific fields
    task_stats: dict[CreditPhase, WorkerPhaseTaskStats] = Field(
        ...,
        description="Stats for the tasks that have been sent to the worker, keyed by the credit phase",
    )

    @property
    def total_tasks(self) -> int:
        """The total number of tasks that have been sent to the worker."""
        return sum(task_stats.total for task_stats in self.task_stats.values())

    @property
    def completed_tasks(self) -> int:
        """The number of tasks that have been completed by the worker."""
        return sum(task_stats.completed for task_stats in self.task_stats.values())

    @property
    def failed_tasks(self) -> int:
        """The number of tasks that have failed by the worker."""
        return sum(task_stats.failed for task_stats in self.task_stats.values())

    @property
    def in_progress_tasks(self) -> int:
        """The number of tasks that are currently in progress by the worker."""
        return sum(task_stats.in_progress for task_stats in self.task_stats.values())

    @property
    def error_rate(self) -> float:
        """The error rate of the worker."""
        if self.total_tasks == 0:
            return 0
        return self.failed_tasks / self.total_tasks
