# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


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

    @property
    def in_progress(self) -> int:
        """The number of tasks that are currently in progress.

        This is the total number of tasks sent to the worker minus the number of failed and successfully completed tasks.
        """
        return self.total - self.completed - self.failed
