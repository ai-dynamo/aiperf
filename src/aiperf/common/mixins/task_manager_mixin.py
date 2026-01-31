# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin
from aiperf.common.utils import yield_to_event_loop

if TYPE_CHECKING:
    from aiperf.common.protocols import TaskManagerProtocol


class TaskManagerMixin(AIPerfLoggerMixin):
    """Mixin to manage a set of async tasks, and provide background task loop capabilities.
    Can be used standalone, but it is most useful as part of the :class:`AIPerfLifecycleMixin`
    mixin, where the lifecycle methods are automatically integrated with the task manager.
    """

    def __init__(self, **kwargs):
        self.tasks: set[asyncio.Task] = set()
        super().__init__(**kwargs)

    def execute_async(self, coro: Coroutine) -> asyncio.Task:
        """Create a task from a coroutine and add it to the set of tasks, and return immediately.
        The task will be automatically cleaned up when it completes.
        """
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def wait_for_tasks(self) -> list[BaseException | None]:
        """Wait for all current tasks to complete."""
        return await asyncio.gather(*list(self.tasks), return_exceptions=True)

    async def cancel_all_tasks(
        self, timeout: float = Environment.SERVICE.TASK_CANCEL_TIMEOUT_SHORT
    ) -> None:
        """Cancel all tasks in the set and wait for up to timeout seconds for them to complete.

        Args:
            timeout: The timeout to wait for the tasks to complete.
        """
        if not self.tasks:
            return

        task_list = list(self.tasks)
        for task in task_list:
            task.cancel()

    def start_background_task(
        self,
        method: Callable,
        interval: float | Callable[[TaskManagerProtocol], float] | None = None,
        immediate: bool = False,
        stop_on_error: bool = False,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Run a task in the background, in a loop until cancelled."""
        self.execute_async(
            self._background_task_loop(
                method, interval, immediate, stop_on_error, stop_event
            )
        )

    async def _background_task_loop(
        self,
        method: Callable,
        interval: float | Callable[[TaskManagerProtocol], float] | None = None,
        immediate: bool = False,
        stop_on_error: bool = False,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Run a background task in a loop until cancelled.

        Args:
            method: The method to run as a background task.
            interval: The interval to run the task in seconds. Can be a callable that returns the interval, and will be called with 'self' as the argument.
            immediate: If True, run the task immediately on start, otherwise wait for the interval first.
            stop_on_error: If True, stop the task on any exception, otherwise log and continue.
        """
        while stop_event is None or not stop_event.is_set():
            try:
                if interval is None or immediate:
                    await yield_to_event_loop()
                    # Reset immediate flag for next iteration otherwise we will not sleep
                    immediate = False
                else:
                    sleep_time = interval(self) if callable(interval) else interval
                    await asyncio.sleep(sleep_time)

                if inspect.iscoroutinefunction(method):
                    await method()
                else:
                    await asyncio.to_thread(method)

                if interval is None:
                    break
            except asyncio.CancelledError:
                self.debug(f"Background task {method.__name__} cancelled")
                break
            except Exception as e:
                self.exception(f"Error in background task {method.__name__}: {e}")
                if stop_on_error:
                    self.exception(
                        f"Background task {method.__name__} stopped due to error"
                    )
                    break
                # Give some time to recover, just in case
                await asyncio.sleep(0.001)
