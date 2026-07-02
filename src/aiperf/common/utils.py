# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
import multiprocessing as mp
import os
import sys
import threading
import types
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any

import orjson

from aiperf.common import aiperf_logger
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.exceptions import AIPerfMultiError

_logger = AIPerfLogger(__name__)


def is_tty() -> bool:
    """Check if stdout is connected to an interactive terminal."""
    return sys.stdout is not None and getattr(sys.stdout, "isatty", lambda: False)()


async def call_all_functions_self(
    self_: object, funcs: list[Callable], *args, **kwargs
) -> None:
    """Call all functions in the list with the given name.

    Args:
        obj: The object to call the functions on.
        func_names: The names of the functions to call.
        *args: The arguments to pass to the functions.
        **kwargs: The keyword arguments to pass to the functions.

    Raises:
        AIPerfMultiError: If any of the functions raise an exception.
    """

    exceptions = []
    for func in funcs:
        try:
            if inspect.iscoroutinefunction(func):
                await func(self_, *args, **kwargs)
            else:
                func(self_, *args, **kwargs)
        except Exception as e:
            _logger.exception(
                f"Error calling function {func.__name__} on {self_.__class__.__name__}: {e!r}"
            )
            exceptions.append(e)

    if len(exceptions) > 0:
        raise AIPerfMultiError("Errors calling functions", exceptions)


async def call_all_functions(funcs: list[Callable], *args, **kwargs) -> None:
    """Call all functions in the list with the given name.

    Args:
        obj: The object to call the functions on.
        func_names: The names of the functions to call.
        *args: The arguments to pass to the functions.
        **kwargs: The keyword arguments to pass to the functions.

    Raises:
        AIPerfMultiError: If any of the functions raise an exception.
    """

    exceptions = []
    for func in funcs:
        try:
            if inspect.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
        except Exception as e:
            _logger.exception(f"Error calling function {func.__name__}: {e!r}")
            exceptions.append(e)

    if len(exceptions) > 0:
        raise AIPerfMultiError("Errors calling functions", exceptions)


def load_json_str(
    json_str: str | bytes, func: Callable = lambda x: x
) -> dict[str, Any]:
    """
    Deserializes JSON encoded string or bytes into Python object.

    Args:
      - json_str: string or bytes
          JSON encoded string or bytes
      - func: callable
          A function that takes deserialized JSON object. This can be used to
          run validation checks on the object. Defaults to identity function.
    """
    try:
        # Note: orjson may not parse JSON the same way as Python's standard json library,
        # notably being stricter on UTF-8 conformance.
        # Refer to https://github.com/ijl/orjson?tab=readme-ov-file#str for details.
        return func(orjson.loads(json_str))
    except orjson.JSONDecodeError as e:
        raw = (
            json_str[:200]
            if isinstance(json_str, str)
            else json_str[:200].decode("utf-8", errors="replace")
        )
        snippet = raw + ("..." if len(json_str) > 200 else "")
        _logger.warning(f"Failed to parse JSON string: '{snippet}' - {e!r}")
        raise


@types.coroutine
def yield_to_event_loop() -> Awaitable[None]:
    """Yield to the event loop. This forces the current coroutine to yield and allow
    other coroutines to run, preventing starvation. Use this when you do not want to
    delay your coroutine via sleep, but still want to allow other coroutines to run if
    there is a potential for an infinite loop.

    NOTE: This still must be called using `await`. It is not defined as `async def` because it uses
    a lower-level asyncio technique than the overhead of calling `asyncio.sleep(0)` in a nested coroutine.
    """
    yield


def compute_time_ns(
    start_time_ns: int, start_perf_ns: int, perf_ns: int | None
) -> int | None:
    """Convert a perf_ns timestamp to a wall clock time_ns timestamp by
    computing the absolute duration in perf_ns (perf_ns - start_perf_ns) and adding it to the start_time_ns.

    Args:
        start_time_ns: The wall clock start time in nanoseconds (time.time_ns).
        start_perf_ns: The start perf time in nanoseconds (perf_counter_ns).
        perf_ns: The perf time in nanoseconds to convert to time_ns (perf_counter_ns).

    Returns:
        The perf_ns converted to time_ns, or None if the perf_ns is None.
    """
    if perf_ns is None:
        return None
    if perf_ns < start_perf_ns:
        raise ValueError(f"perf_ns {perf_ns} is before start_perf_ns {start_perf_ns}")
    return start_time_ns + (perf_ns - start_perf_ns)


def _set_daemon(daemon: bool) -> None:
    """Set the daemon flag on the current process."""
    try:
        mp.current_process().daemon = daemon
    except AssertionError:
        # Fallback to the internal _config dict when assertions are enabled.
        mp.current_process()._config["daemon"] = daemon


# Reentrancy/thread-safety for allow_daemon_children: the daemon flag is
# process-global, but callers may enter concurrently — e.g. LCB grading pushes
# codegen_metrics to asyncio.to_thread so multiple grade() calls fan out at
# once. A naive clear/restore lets one caller restore daemon=True while another
# is still spawning workers, intermittently reintroducing the very
# "daemonic processes are not allowed to have children" crash. A lock-protected
# depth counter clears on the first entry and restores only when the last
# concurrent caller exits.
_daemon_override_lock = threading.Lock()
_daemon_override_depth = 0
_daemon_override_was_daemon = False


@contextmanager
def allow_daemon_children() -> Iterator[None]:
    """Temporarily clear the current process's daemon flag (reentrant, thread-safe).

    Python's multiprocessing refuses to spawn children from daemon
    processes, and AIPerf services run as daemons (see
    ``multiprocess_service_manager.py``: every service is spawned with
    ``daemon=True``). Any code that fans out with ``ProcessPoolExecutor``
    or ``multiprocessing.Pool`` from inside a service must run under this
    context, or it raises ``AssertionError: daemonic processes are not
    allowed to have children``.

    Safe under concurrent/nested use within a process: the flag is cleared
    while any caller is active and restored to its original value only when the
    outermost/last caller exits. The lock is held only around the flag
    mutation, not during the wrapped work, so concurrent pools still run in
    parallel.
    """
    global _daemon_override_depth, _daemon_override_was_daemon
    with _daemon_override_lock:
        if _daemon_override_depth == 0:
            _daemon_override_was_daemon = mp.current_process().daemon
            if _daemon_override_was_daemon:
                _set_daemon(False)
        _daemon_override_depth += 1
    try:
        yield
    finally:
        with _daemon_override_lock:
            _daemon_override_depth -= 1
            if _daemon_override_depth == 0 and _daemon_override_was_daemon:
                _set_daemon(True)


# This is used to identify the source file of the call_all_functions function
# in the AIPerfLogger class to skip it when determining the caller.
# NOTE: Using similar logic to logging._srcfile
_srcfile = os.path.normcase(call_all_functions.__code__.co_filename)
aiperf_logger._ignored_files.append(_srcfile)
