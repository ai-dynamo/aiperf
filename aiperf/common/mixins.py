# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
from collections.abc import Coroutine

from aiperf.common.constants import TASK_CANCEL_TIMEOUT_SHORT


class AsyncTaskManagerMixin:
    """Mixin to manage a set of async tasks."""

    def __init__(self):
        super().__init__()
        self.tasks: set[asyncio.Task] = set()

    def execute_async(self, coro: Coroutine) -> asyncio.Task:
        """Create a task from a coroutine and add it to the set of tasks, and return immediately.
        The task will be automatically cleaned up when it completes.
        """
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def cancel_all_tasks(
        self, timeout: float = TASK_CANCEL_TIMEOUT_SHORT
    ) -> None:
        """Cancel all tasks in the set and wait for up to timeout seconds for them to complete.

        Args:
            timeout: The timeout to wait for the tasks to complete.
        """
        if not self.tasks:
            return

        for task in list(self.tasks):
            task.cancel()

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.gather(*self.tasks), timeout=timeout)

        # Clear the tasks set after cancellation to avoid memory leaks
        self.tasks.clear()
