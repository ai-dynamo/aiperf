# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import msgspec

from aiperf.common.models.base_models import PydanticStructMixin


class WorkerTaskStats(
    PydanticStructMixin,
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
):
    """Stats for the tasks that have been sent to the worker.

    Mutable accumulator: ``task_finished`` increments ``failed`` or
    ``completed`` in place, so the struct intentionally omits ``frozen``.
    """

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
