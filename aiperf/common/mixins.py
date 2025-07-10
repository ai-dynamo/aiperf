# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import time
from collections.abc import Coroutine

import psutil

from aiperf.common.constants import BYTES_PER_MIB, TASK_CANCEL_TIMEOUT_SHORT
from aiperf.common.health_models import CPUTimes, CtxSwitches, ProcessHealth


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

    async def stop(self) -> None:
        """Stop all tasks in the set and wait for them to complete."""
        await self.cancel_all_tasks()

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
            if not task.done():
                task.cancel()

        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(
                asyncio.gather(*self.tasks, return_exceptions=True), timeout=timeout
            )

        # Clear the tasks set after cancellation to avoid memory leaks
        self.tasks.clear()


class ProcessHealthMixin:
    """Mixin to provide process health information."""

    def __init__(self):
        super().__init__()
        # Initialize process-specific CPU monitoring
        self.process: psutil.Process = psutil.Process()
        self.process.cpu_percent()  # throw away the first result (will be 0)
        self.create_time: float = self.process.create_time()

        self.process_health: ProcessHealth | None = None
        self.previous: ProcessHealth | None = None

    def get_process_health(self) -> ProcessHealth:
        """Get the process health information for the current process."""

        # Get process-specific CPU and memory usage
        raw_cpu_times = self.process.cpu_times()
        cpu_times = CPUTimes(
            user=raw_cpu_times[0],
            system=raw_cpu_times[1],
            iowait=raw_cpu_times[4] if len(raw_cpu_times) > 4 else 0.0,  # type: ignore
        )

        self.previous = self.process_health

        self.process_health = ProcessHealth(
            pid=self.process.pid,
            create_time=self.create_time,
            uptime=time.time() - self.create_time,
            cpu_usage=self.process.cpu_percent(),
            memory_usage=self.process.memory_info().rss / BYTES_PER_MIB,
            io_counters=self.process.io_counters(),
            cpu_times=cpu_times,
            num_ctx_switches=CtxSwitches(*self.process.num_ctx_switches()),
            num_threads=self.process.num_threads(),
        )
        return self.process_health
