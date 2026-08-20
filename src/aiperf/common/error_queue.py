# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backchannel multiprocessing.Queue for reporting errors from child processes.

Unlike the log queue (which is only active with the Dashboard UI), the error
queue is always created when using multiprocessing. Child processes put
serialized ``ExitErrorInfo`` payloads onto the queue when they encounter
errors, and the parent process drains it during shutdown to surface subprocess
failures that would otherwise vanish with the child.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import queue
import threading
from typing import TYPE_CHECKING, Protocol, TypeAlias

from pydantic import ValidationError

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import Environment

ErrorQueue: TypeAlias = multiprocessing.Queue

if TYPE_CHECKING:
    from aiperf.common.models.error_models import ExitErrorInfo

_logger = AIPerfLogger(__name__)
_global_error_queue: ErrorQueue | None = None
_error_queue_lock = threading.Lock()


def get_global_error_queue() -> ErrorQueue:
    """Get the global error queue, creating it on first call.

    Thread-safe singleton pattern using double-checked locking.

    Example:
        ```python
        manager = SubprocessManager(run=run, error_queue=get_global_error_queue())
        ```
    """
    global _global_error_queue
    if _global_error_queue is None:
        with _error_queue_lock:
            if _global_error_queue is None:
                from aiperf.common.mp_context import get_mp_context

                _global_error_queue = get_mp_context().Queue(
                    maxsize=Environment.SERVICE.ERROR_QUEUE_MAXSIZE
                )
    return _global_error_queue


async def cleanup_global_error_queue() -> None:
    """Close the global error queue to prevent semaphore leaks.

    Should be called during shutdown after draining remaining errors.
    Thread-safe.
    """
    global _global_error_queue
    with _error_queue_lock:
        if _global_error_queue is not None:
            try:
                _global_error_queue.close()
                await asyncio.wait_for(
                    asyncio.to_thread(_global_error_queue.join_thread), timeout=1.0
                )
                _logger.debug("Cleaned up global error queue")
            except (TimeoutError, OSError, ValueError) as e:
                # OSError from close()/join_thread on dead handles; ValueError from closed queue.
                _logger.debug(f"Error cleaning up error queue: {e}")
            finally:
                from aiperf.common.resource_tracker import unregister_queue_semaphores

                unregister_queue_semaphores(_global_error_queue)
                _global_error_queue = None


def drain_error_queue(error_queue: ErrorQueue) -> list[ExitErrorInfo]:
    """Drain all pending errors from the queue without blocking.

    Items that fail to deserialize are logged at debug level and skipped, so a
    single malformed payload cannot hide the rest of the child's errors.

    Args:
        error_queue: The multiprocessing error queue to drain.

    Returns:
        List of ExitErrorInfo reported by child processes.
    """
    from aiperf.common.models.error_models import ExitErrorInfo

    errors: list[ExitErrorInfo] = []
    while True:
        try:
            data = error_queue.get_nowait()
        except queue.Empty:
            break
        if isinstance(data, ExitErrorInfo):
            errors.append(data)
            continue
        try:
            errors.append(ExitErrorInfo.model_validate(data))
        except (ValidationError, TypeError) as e:
            _logger.debug(f"Failed to deserialize error queue item: {e}")
    return errors


def report_errors(error_queue: ErrorQueue, errors: list[ExitErrorInfo]) -> None:
    """Put accumulated service errors onto the error queue from a child process.

    Non-blocking: silently drops errors if the queue is full, because a child
    that is already exiting on error must never block on the backchannel.

    Args:
        error_queue: The multiprocessing error queue.
        errors: List of ExitErrorInfo accumulated by the service lifecycle.
    """
    for error_info in errors:
        try:
            error_queue.put_nowait(error_info.model_dump(mode="json"))
        except queue.Full:
            break
        except (OSError, ValueError) as e:
            # Queue handle closed/broken during shutdown; drop silently per docstring contract.
            _logger.debug(f"Failed to enqueue error info: {e}")


# ---------------------------------------------------------------------------
# Consumer-side collector
# ---------------------------------------------------------------------------


class _ErrorLogger(Protocol):
    """Minimal logging surface required by ``ErrorCollector``."""

    def error(self, message: str) -> None: ...


class ErrorCollector:
    """Collects errors reported by child processes via the error queue.

    Created by any component that spawns subprocesses (SystemController,
    WorkerManager). Provides the queue to pass to ``SubprocessManager`` and a
    ``drain_into`` method to collect errors during shutdown.

    Example:
        ```python
        collector = ErrorCollector(logger=self, exit_errors=self._exit_errors)
        manager = SubprocessManager(run=run, error_queue=collector.error_queue)
        ...
        collector.drain_into()
        ```
    """

    def __init__(
        self,
        logger: _ErrorLogger,
        exit_errors: list[ExitErrorInfo],
    ) -> None:
        """Initialize the collector.

        Args:
            logger: Object with an ``error()`` method for logging drained errors.
            exit_errors: List to extend with drained errors (typically
                ``_exit_errors`` from ``AIPerfLifecycleMixin``).
        """
        self._error_queue = get_global_error_queue()
        self._logger = logger
        self._exit_errors = exit_errors

    @property
    def error_queue(self) -> ErrorQueue:
        """The multiprocessing queue for subprocess error reporting."""
        return self._error_queue

    def drain_into(self) -> list[ExitErrorInfo]:
        """Drain subprocess errors, log each one, and append to ``exit_errors``.

        Returns:
            The list of errors that were drained.
        """
        errors = drain_error_queue(self._error_queue)
        for err in errors:
            self._logger.error(
                f"Subprocess error from {err.service_id}: {err.error_details.message}"
            )
        self._exit_errors.extend(errors)
        return errors
