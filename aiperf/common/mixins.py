# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import os
import time
from collections.abc import Callable, Coroutine
from typing import Protocol, runtime_checkable

import psutil

from aiperf.common import aiperf_logger
from aiperf.common.aiperf_logger import (
    _CRITICAL,
    _DEBUG,
    _ERROR,
    _INFO,
    _NOTICE,
    _SUCCESS,
    _TRACE,
    _WARNING,
    AIPerfLogger,
)
from aiperf.common.constants import BYTES_PER_MIB, TASK_CANCEL_TIMEOUT_SHORT
from aiperf.common.health_models import CPUTimes, CtxSwitches, ProcessHealth


class AsyncTaskManagerMixin:
    """Mixin to manage a set of async tasks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

        with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(
                asyncio.gather(*self.tasks, return_exceptions=True), timeout=timeout
            )

        # Clear the tasks set after cancellation to avoid memory leaks
        self.tasks.clear()


@runtime_checkable
class AsyncTaskManagerProtocol(Protocol):
    """Protocol to manage a set of async tasks."""

    def execute_async(self, coro: Coroutine) -> asyncio.Task:
        """Create a task from a coroutine and add it to the set of tasks, and return immediately.
        The task will be automatically cleaned up when it completes.
        """
        ...

    async def stop(self) -> None:
        """Stop all tasks in the set and wait for them to complete."""

    async def cancel_all_tasks(
        self, timeout: float = TASK_CANCEL_TIMEOUT_SHORT
    ) -> None:
        """Cancel all tasks in the set and wait for up to timeout seconds for them to complete.

        Args:
            timeout: The timeout to wait for the tasks to complete.
        """


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


class AIPerfLoggerMixin:
    """Mixin to provide lazy evaluated logging for f-strings.

    This mixin provides a logger with lazy evaluation support for f-strings,
    and direct log functions for all standard and custom logging levels.

    see :class:`AIPerfLogger` for more details.

    Usage:
        class MyClass(AIPerfLoggerMixin):
            def __init__(self):
                super().__init__()
                self.trace(lambda: f"Processing {item} of {count} ({item / count * 100}% complete)")
                self.info("Simple string message")
                self.debug(lambda i=i: f"Binding loop variable: {i}")
                self.warning("Warning message: %s", "legacy support")
                self.success("Benchmark completed successfully")
                self.notice("Warmup has completed")
                self.exception(f"Direct f-string usage: {e}")
    """

    def __init__(self, logger_name: str | None = None, **kwargs) -> None:
        super().__init__()
        self.logger = AIPerfLogger(logger_name or self.__class__.__name__)
        self._log = self.logger._log
        self.is_enabled_for = self.logger.is_enabled_for

    def log(
        self, level: int, message: str | Callable[..., str], *args, **kwargs
    ) -> None:
        """Log a message at a specified level with lazy evaluation."""
        if self.is_enabled_for(level):
            self._log(level, message, *args, **kwargs)

    def trace(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a trace message with lazy evaluation."""
        if self.is_enabled_for(_TRACE):
            self._log(_TRACE, message, *args, **kwargs)

    def debug(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a debug message with lazy evaluation."""
        if self.is_enabled_for(_DEBUG):
            self._log(_DEBUG, message, *args, **kwargs)

    def info(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log an info message with lazy evaluation."""
        if self.is_enabled_for(_INFO):
            self._log(_INFO, message, *args, **kwargs)

    def notice(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a notice message with lazy evaluation."""
        if self.is_enabled_for(_NOTICE):
            self._log(_NOTICE, message, *args, **kwargs)

    def warning(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a warning message with lazy evaluation."""
        if self.is_enabled_for(_WARNING):
            self._log(_WARNING, message, *args, **kwargs)

    def success(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a success message with lazy evaluation."""
        if self.is_enabled_for(_SUCCESS):
            self._log(_SUCCESS, message, *args, **kwargs)

    def error(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log an error message with lazy evaluation."""
        if self.is_enabled_for(_ERROR):
            self._log(_ERROR, message, *args, **kwargs)

    def exception(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log an exception message with lazy evaluation."""
        if self.is_enabled_for(_ERROR):
            self._log(_ERROR, message, *args, exc_info=True, **kwargs)

    def critical(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a critical message with lazy evaluation."""
        if self.is_enabled_for(_CRITICAL):
            self._log(_CRITICAL, message, *args, **kwargs)


# Add this file to the list of ignored files to avoid this file from being the source of the log messages
# in the AIPerfLogger class to skip it when determining the caller.
# NOTE: Using similar logic to logging._srcfile
_srcfile = os.path.normcase(AIPerfLoggerMixin.info.__code__.co_filename)
aiperf_logger._ignored_files.append(_srcfile)
