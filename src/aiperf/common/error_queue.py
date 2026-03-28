# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backchannel multiprocessing.Queue for reporting errors from child processes.

Unlike the log_queue (which is only active with Dashboard UI), the error_queue
is always created when using multiprocessing. Child processes put ExitErrorInfo
dicts onto the queue when they encounter errors, and the parent process drains
it during shutdown to surface subprocess errors.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import queue
import threading
from typing import TYPE_CHECKING, Protocol, TypeAlias

from aiperf.common.aiperf_logger import AIPerfLogger

ErrorQueue: TypeAlias = multiprocessing.Queue

if TYPE_CHECKING:
    from aiperf.common.models.error_models import ExitErrorInfo

_logger = AIPerfLogger(__name__)
_global_error_queue: ErrorQueue | None = None
_error_queue_lock = threading.Lock()

_ERROR_QUEUE_MAXSIZE = 256


def get_global_error_queue() -> ErrorQueue:
    """Get the global error queue. Creates a new queue on first call.

    Thread-safe singleton pattern using double-checked locking.
    """
    global _global_error_queue
    if _global_error_queue is None:
        with _error_queue_lock:
            if _global_error_queue is None:
                from aiperf.common.subprocess_manager import get_mp_context

                _global_error_queue = get_mp_context().Queue(
                    maxsize=_ERROR_QUEUE_MAXSIZE
                )
    return _global_error_queue


async def cleanup_global_error_queue() -> None:
    """Clean up the global error queue to prevent semaphore leaks.

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
            except Exception as e:
                _logger.debug(f"Error cleaning up error queue: {e}")
            finally:
                from aiperf.common.resource_tracker import unregister_queue_semaphores

                unregister_queue_semaphores(_global_error_queue)
                _global_error_queue = None


def drain_error_queue(
    error_queue: ErrorQueue,
) -> list[ExitErrorInfo]:
    """Drain all pending errors from the queue without blocking.

    Returns:
        List of ExitErrorInfo from child processes.
    """
    from aiperf.common.models.error_models import ExitErrorInfo

    errors: list[ExitErrorInfo] = []
    while True:
        try:
            data = error_queue.get_nowait()
            errors.append(ExitErrorInfo.model_validate(data))
        except queue.Empty:
            break
        except Exception as e:
            _logger.debug(f"Failed to deserialize error queue item: {e}")
    return errors


def report_errors(
    error_queue: ErrorQueue,
    errors: list[ExitErrorInfo],
) -> None:
    """Put accumulated service errors onto the error queue from a child process.

    Non-blocking: silently drops errors if the queue is full.

    Args:
        error_queue: The multiprocessing error queue.
        errors: List of ExitErrorInfo accumulated by the service lifecycle.
    """
    for error_info in errors:
        try:
            error_queue.put_nowait(error_info.model_dump())
        except queue.Full:
            break
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Consumer-side collector
# ---------------------------------------------------------------------------


class _ErrorLogger(Protocol):
    def error(self, message: str) -> None: ...


class ErrorCollector:
    """Collects errors reported by child processes via the error queue.

    Created by any component that spawns subprocesses (SystemController,
    WorkerGroupManager). Provides the queue to pass to SubprocessManager and
    a drain_into method to collect errors during shutdown.

    Args:
        logger: Object with an error() method for logging drained errors.
        exit_errors: List to extend with drained errors (typically _exit_errors
            from AIPerfLifecycleMixin).
    """

    def __init__(
        self,
        logger: _ErrorLogger,
        exit_errors: list[ExitErrorInfo],
    ) -> None:
        self._error_queue = get_global_error_queue()
        self._logger = logger
        self._exit_errors = exit_errors

    @property
    def error_queue(self) -> ErrorQueue:
        """The multiprocessing queue for subprocess error reporting."""
        return self._error_queue

    def drain_into(self) -> list[ExitErrorInfo]:
        """Drain subprocess errors, log each one, and append to exit_errors.

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
