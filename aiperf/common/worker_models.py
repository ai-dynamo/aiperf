# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from functools import cached_property
from typing import Literal

from pydantic import Field

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.health_models import ProcessHealth
from aiperf.common.messages import BaseServiceMessage
from aiperf.common.pydantic_utils import AIPerfBaseModel


class WorkerPhaseTaskStats(AIPerfBaseModel):
    """Stats for the tasks that have been sent to the worker for a given credit phase."""

    total: int = Field(
        default=0,
        description="The total number of tasks that have been sent to the worker. "
        "Not all tasks will be completed.",
    )
    failed: int = Field(
        default=0,
        description="The number of tasks that returned an error",
    )
    completed: int = Field(
        default=0,
        description="The number of tasks that were completed successfully",
    )

    @cached_property
    def in_progress(self) -> int:
        """The number of tasks that are currently in progress.

        This is the total number of tasks sent to the worker minus the number of failed and successfully completed tasks.
        """
        return self.total - self.completed - self.failed


class WorkerHealthMessage(BaseServiceMessage):
    """Message for a worker health check."""

    message_type: Literal[MessageType.WORKER_HEALTH] = MessageType.WORKER_HEALTH

    # override request_ns to be auto-filled if not provided
    request_ns: int = Field(  # type: ignore
        default_factory=time.time_ns,
        description="Timestamp of the request",
    )

    process: ProcessHealth = Field(..., description="The health of the worker process")

    # Worker specific fields
    task_stats: dict[CreditPhase, WorkerPhaseTaskStats] = Field(
        ...,
        description="Stats for the tasks that have been sent to the worker, keyed by the credit phase",
    )
