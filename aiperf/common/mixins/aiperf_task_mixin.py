# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import inspect

from aiperf.common.hooks import (
    AIPerfHook,
    AIPerfTaskHook,
    on_init,
    on_stop,
    supports_hooks,
)
from aiperf.common.mixins.async_task_manager_mixin import AsyncTaskManagerMixin
from aiperf.common.mixins.hooks_mixin import HooksMixin


@supports_hooks(
    AIPerfTaskHook.AIPERF_TASK,
    AIPerfHook.ON_INIT,
    AIPerfHook.ON_START,
    AIPerfHook.ON_STOP,
)
class AIPerfTaskMixin(HooksMixin, AsyncTaskManagerMixin):
    """Mixin to add aiperf_task support to a class.

    It hooks into the :meth:`HooksMixin.on_init` and :meth:`HooksMixin.on_stop` hooks to
    start and stop the tasks.
    """

    # TODO: This is somewhat deprecated in favor of the lifecycle mixin.

    # TODO: Once we create a Mixin for `self.stop_event`, we can avoid
    # having the user to call `while not self.stop_event.is_set()`

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def initialize(self) -> None:
        """Initialize the task."""
        await self.run_hooks(AIPerfHook.ON_INIT)

    async def start(self) -> None:
        """Start the task."""
        await self.run_hooks(AIPerfHook.ON_START)

    async def stop(self) -> None:
        """Stop the task."""
        await self.run_hooks(AIPerfHook.ON_STOP)

    # TODO: Should this be on_start?
    @on_init
    async def _start_tasks(self):
        """Start all the registered tasks in the background."""
        for hook in self.get_hooks(AIPerfTaskHook.AIPERF_TASK):
            if inspect.iscoroutinefunction(hook):
                self.execute_async(hook())
            else:
                self.execute_async(asyncio.to_thread(hook))

    @on_stop
    async def _stop_tasks(self):
        """Stop all the background tasks. This will wait for all the tasks to complete."""
        await self.cancel_all_tasks()
