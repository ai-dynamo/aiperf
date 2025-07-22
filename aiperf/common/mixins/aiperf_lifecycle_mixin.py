# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import inspect
from collections.abc import Callable

from aiperf.common.exceptions import InvalidStateError
from aiperf.common.hooks import (
    AIPerfHook,
    AIPerfTaskHook,
    on_start,
    on_stop,
    supports_hooks,
)
from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin
from aiperf.common.mixins.async_task_manager_mixin import AsyncTaskManagerMixin
from aiperf.common.mixins.hooks_mixin import HooksMixin


@supports_hooks(
    AIPerfTaskHook.AIPERF_TASK,
    AIPerfTaskHook.AIPERF_AUTO_TASK,
    AIPerfHook.ON_INIT,
    AIPerfHook.ON_START,
    AIPerfHook.ON_STOP,
    AIPerfHook.ON_CLEANUP,
)
class AIPerfLifecycleMixin(HooksMixin, AsyncTaskManagerMixin, AIPerfLoggerMixin):
    """Mixin to add task support to a class. It abstracts away the details of the
    :class:`AIPerfTask` and provides a simple interface for registering and running tasks.
    It hooks into the :meth:`HooksMixin.on_start` and :meth:`HooksMixin.on_stop` hooks to
    start and stop the tasks.
    """

    def __init__(self, **kwargs):
        self.initialized_event: asyncio.Event = asyncio.Event()
        self.started_event: asyncio.Event = asyncio.Event()
        self.stop_requested: asyncio.Event = asyncio.Event()
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.lifecycle_task: asyncio.Task | None = None
        super().__init__(**kwargs)

    def is_initialized(self) -> bool:
        """Check if the lifecycle has been initialized."""
        return self.initialized_event.is_set()

    async def _run_lifecycle(self) -> None:
        """Run the internal lifecycle."""
        # Run all the initialization hooks and set the initialize_event
        await self.run_hooks(AIPerfHook.ON_INIT)
        self.initialized_event.set()

        # Run all the start hooks and set the start_event
        await self.run_hooks_async(AIPerfHook.ON_START)
        self.started_event.set()

        while not self.stop_requested.is_set() and not self.shutdown_event.is_set():
            try:
                # Wait forever until the stop_requested event is set
                await self.stop_requested.wait()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Unhandled exception in lifecycle: %s", e)
                continue

        try:
            # Run all the stop hooks
            await self.run_hooks_async(AIPerfHook.ON_STOP)
        except Exception as e:
            self.logger.exception("Unhandled exception in lifecycle: %s", e)

        try:
            # Run all the cleanup hooks and set the shutdown_event
            await self.run_hooks(AIPerfHook.ON_CLEANUP)
        except Exception as e:
            self.logger.exception("Unhandled exception in lifecycle: %s", e)
        finally:
            self.shutdown_event.set()

        self.trace("Lifecycle finished")

    async def run_async(self) -> None:
        """Start the lifecycle in the background. Will call the :meth:`HooksMixin.on_init` hooks,
        followed by the :meth:`HooksMixin.on_start` hooks. Will return immediately."""
        if self.lifecycle_task is not None:
            raise InvalidStateError("Lifecycle is already running")
        self.lifecycle_task = asyncio.create_task(self._run_lifecycle())

    async def run_and_wait_for_start(self) -> None:
        """Start the lifecycle in the background and wait until the lifecycle is initialized and started.
        Will call the :meth:`HooksMixin.on_init` hooks, followed by the :meth:`HooksMixin.on_start` hooks."""
        if self.lifecycle_task is not None:
            raise InvalidStateError("Lifecycle is already running")
        self.lifecycle_task = asyncio.create_task(self._run_lifecycle())

        await self.initialized_event.wait()
        await self.started_event.wait()

    async def wait_for_initialize(self) -> None:
        """Wait for the lifecycle to be initialized. Will wait until the :meth:`HooksMixin.on_init` hooks have been called.
        Will return immediately if the lifecycle is already initialized."""
        await self.initialized_event.wait()

    async def wait_for_start(self) -> None:
        """Wait for the lifecycle to be started. Will wait until the :meth:`HooksMixin.on_start` hooks have been called.
        Will return immediately if the lifecycle is already started."""
        await self.started_event.wait()

    async def wait_for_shutdown(self) -> None:
        """Wait for the lifecycle to be shutdown. Will wait until the :meth:`HooksMixin.on_stop` hooks have been called.
        Will return immediately if the lifecycle is already shutdown."""
        await self.shutdown_event.wait()

    async def shutdown(self) -> None:
        """Shutdown the lifecycle. Will call the :meth:`HooksMixin.on_stop` hooks,
        followed by the :meth:`HooksMixin.on_cleanup` hooks."""
        self.stop_requested.set()

    @on_start
    async def _start_tasks(self):
        """Start all the registered tasks in the background."""

        # Start all the non-auto tasks
        for hook in self.get_hooks(AIPerfTaskHook.AIPERF_TASK):
            if inspect.iscoroutinefunction(hook):
                self.execute_async(hook())
            else:
                self.execute_async(asyncio.to_thread(hook))

        # Start all the auto tasks
        for hook in self.get_hooks(AIPerfTaskHook.AIPERF_AUTO_TASK):
            interval = getattr(hook, AIPerfTaskHook.AIPERF_AUTO_TASK_INTERVAL, None)
            self.execute_async(self._task_wrapper(hook, interval))

    @on_stop
    async def _stop_tasks(self):
        """Stop all the background tasks. This will wait for all the tasks to complete."""
        await self.cancel_all_tasks()

    async def _task_wrapper(
        self, func: Callable, interval: float | None = None
    ) -> None:
        """Wrapper to run a task in a loop until the stop_requested event is set."""
        while not self.stop_requested.is_set():
            try:
                if inspect.iscoroutinefunction(func):
                    await func()
                else:
                    await asyncio.to_thread(func)
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("Unhandled exception in task: %s", func.__name__)

            if interval is None:
                break
            await asyncio.sleep(interval)
