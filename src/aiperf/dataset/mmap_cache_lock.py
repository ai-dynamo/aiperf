# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-process populate lock for the mmap dataset cache.

Wraps :class:`filelock.FileLock` in an async-friendly context manager that
mirrors HuggingFace ``WeakFileLock``: periodic INFO log while waiting,
SoftFileLock fallback on filesystems without ``flock``, group-writable
lock files so multiple users sharing a cache contend correctly.

Used by :mod:`aiperf.dataset.mmap_cache` to serialize concurrent populates
on the same cache key. Callers should follow the double-checked pattern:
look up the cache, on miss enter this lock, look up again under the lock,
populate only on second miss.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from filelock import FileLock, SoftFileLock, Timeout

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)

LOCK_FILENAME_SUFFIX = ".lock"

# How often to emit an INFO log while blocked waiting for a populate lock.
_LOCK_LOG_EVERY_SECONDS = 10.0
# Default outer timeout for a populate-lock acquire. Long enough that the
# slowest tokenize-and-mmap on a multi-GB trace corpus comfortably finishes
# before a waiter gives up. Override via ``timeout`` kwarg.
_LOCK_DEFAULT_TIMEOUT_S = 1800.0


def _blocking_acquire(
    lock: FileLock | SoftFileLock, timeout: float | None, lock_path: Path
) -> None:
    """Acquire ``lock`` with periodic INFO logs (mirrors HF ``WeakFileLock``).

    Retries the acquire in ``_LOCK_LOG_EVERY_SECONDS`` chunks so a waiter
    prints visible progress messages instead of hanging silently. Raises
    :class:`filelock.Timeout` if ``timeout`` elapses before acquire.
    """
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if timeout is not None and elapsed >= timeout:
            raise Timeout(str(lock_path))
        per_attempt = (
            min(_LOCK_LOG_EVERY_SECONDS, timeout - elapsed)
            if timeout is not None
            else _LOCK_LOG_EVERY_SECONDS
        )
        try:
            lock.acquire(timeout=per_attempt)
            return
        except Timeout:
            _logger.info(
                lambda: (
                    f"Still waiting on mmap-cache populate lock at "
                    f"{lock_path} (elapsed: {time.monotonic() - start:.1f}s)"
                )
            )


@contextlib.asynccontextmanager
async def acquire_cache_lock(
    cache_key: str,
    *,
    cache_dir_resolver: Callable[[], Path],
    timeout: float | None = _LOCK_DEFAULT_TIMEOUT_S,
) -> AsyncIterator[None]:
    """Hold an exclusive cross-process lock for ``cache_key`` populates.

    Use the double-checked pattern: caller looks up the cache, and on miss
    enters this context. The expensive tokenize + populate runs under the
    lock; concurrent processes block on the same key and wake to find the
    cache populated by the winner. Re-lookup MUST happen inside the lock to
    pick up the winner's entry.

    Lock files are created mode 0o664 so multiple users sharing a cache
    directory can contend on the same lock. Falls back to ``SoftFileLock``
    if the underlying filesystem does not support ``flock`` (some NFS
    configurations). The acquire runs on a worker thread so the event loop
    is not blocked.

    ``cache_dir_resolver`` is a zero-arg callable that returns the cache
    directory. It is injected (rather than imported) to avoid a circular
    import with :mod:`aiperf.dataset.mmap_cache`.
    """
    base = cache_dir_resolver()
    base.mkdir(parents=True, exist_ok=True)
    lock_path = base / f"{cache_key}{LOCK_FILENAME_SUFFIX}"
    # ``thread_local=False`` is required: the acquire runs on an
    # ``asyncio.to_thread`` worker, but the release fires from the finally
    # below on whatever worker the event loop picks next. With the default
    # (thread-local counter) the release runs on a thread whose TLS doesn't
    # know about the acquire and silently no-ops, leaving the OS lock held
    # forever.
    lock: FileLock | SoftFileLock = FileLock(
        str(lock_path), mode=0o664, thread_local=False
    )
    try:
        await asyncio.to_thread(_blocking_acquire, lock, timeout, lock_path)
    except NotImplementedError as e:
        if "use SoftFileLock instead" not in str(e):
            raise
        _logger.warning(
            lambda: (
                f"Filesystem at {lock_path} does not support flock; "
                f"falling back to SoftFileLock (less robust on crash)."
            )
        )
        lock = SoftFileLock(str(lock_path), thread_local=False)
        await asyncio.to_thread(_blocking_acquire, lock, timeout, lock_path)
    try:
        yield
    finally:
        try:
            await asyncio.to_thread(lock.release)
        except OSError:
            _logger.debug(
                lambda: f"Best-effort release of mmap-cache lock {lock_path} failed."
            )
