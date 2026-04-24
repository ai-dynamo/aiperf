# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from typing import ClassVar

from pydantic import ConfigDict


@dataclass(slots=True, kw_only=True)
class WorkerTaskStats:
    """Stats for the tasks that have been sent to the worker.

    Mutable slotted dataclass — shared type for msgspec envelopes
    (``WorkerHealthMessage.task_stats``) and Pydantic
    (``WorkerStats.task_stats`` via ``WorkersResponse``). ``task_finished``
    mutates in place, so the dataclass is not frozen.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    total: int = 0
    failed: int = 0
    completed: int = 0

    def task_finished(self, valid: bool) -> None:
        """Increment the task stats based on success or failure."""
        if not valid:
            self.failed += 1
        else:
            self.completed += 1

    @property
    def in_progress(self) -> int:
        """The number of tasks that are currently in progress.

        This is the total number of tasks sent to the worker minus the number of failed and successfully completed tasks.
        """
        return self.total - self.completed - self.failed
