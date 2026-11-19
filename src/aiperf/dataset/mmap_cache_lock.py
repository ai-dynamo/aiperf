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
import os
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
# Substring of the ``NotImplementedError`` message raised by filelock's
# ``UnixFileLock`` when ``fcntl.flock`` is unavailable (NFS without flock,
# some FUSE mounts). Present in filelock 3.14–3.23; 3.13.x typo'd it as
# ``"user SoftFileLock instead"``; filelock>=3.24 auto-mutates to SoftFileLock
# instead of raising, so this branch is only hit on the mid-range of
# ``filelock>=3.13``. A test asserts this constant so a future filelock
# upgrade that drops the string fails loudly.
_FLOCK_UNSUPPORTED_HINT = "use SoftFileLock instead"
# Group-writable so multiple users sharing a cache dir can contend on the
# same lock file under a restrictive umask (common on shared NFS clusters).
# FileLock applies this via fchmod; SoftFileLock only passes it to os.open
# (umask-masked), so callers must chmod after SoftFileLock acquire.
_LOCK_FILE_MODE = 0o664


def _ensure_lock_file_mode(lock_path: Path) -> None:
    """Force ``_LOCK_FILE_MODE`` on ``lock_path`` (SoftFileLock is umask-masked)."""
    try:
        os.chmod(lock_path, _LOCK_FILE_MODE)
    except OSError:
        _logger.debug(
            lambda: f"Best-effort chmod of mmap-cache lock {lock_path} failed."
        )


def _blocking_acquire(
    lock: FileLock | SoftFileLock,
    timeout: float | None,
    lock_path: Path,
    cache_complete_check: Callable[[], bool] | None = None,
) -> bool:
    """Acquire ``lock`` with periodic INFO logs (mirrors HF ``WeakFileLock``).

    Retries the acquire in ``_LOCK_LOG_EVERY_SECONDS`` chunks so a waiter
    prints visible progress messages instead of hanging silently. Raises
    :class:`filelock.Timeout` if ``timeout`` elapses before acquire.

    If ``cache_complete_check`` is provided and returns True before or
    between acquire attempts, bails out without acquiring the lock and
    returns False — the cache has been populated by some other process
    and the caller's re-check-under-lock will see the HIT regardless of
    whether we hold the lock. This bypass is what unwedges waiters when
    a populator was SIGKILLed (slurm scancel, OOM-kill, container exit)
    between writing the cache and releasing the lock, leaving the .lock
    file as a tombstone on a shared NFS mount.

    Returns:
        True when the lock was acquired; False when the cache-complete
        bypass triggered.
    """
    start = time.monotonic()
    while True:
        if cache_complete_check is not None and cache_complete_check():
            _logger.info(
                lambda: (
                    f"Cache already populated while waiting on lock at "
                    f"{lock_path}; bypassing acquire (elapsed: "
                    f"{time.monotonic() - start:.1f}s)."
                )
            )
            return False
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
            return True
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
    # Cache-complete bypass: mmap_cache.populate writes manifest.json LAST
    # inside a tmp directory and then atomically renames the whole tmp dir
    # into <cache_dir>/<cache_key>. So the presence of
    # <cache_dir>/<cache_key>/manifest.json implies a complete cache entry.
    # We use it as the bypass signal both before attempting to acquire and
    # between retries; this prevents a stale .lock file (left behind by a
    # SIGKILLed populator on shared NFS) from wedging every subsequent
    # waiter on what is otherwise a fully populated cache. Caller's
    # re-check-under-lock path validates compressed/version match, file
    # existence, etc., so this best-effort signal is conservative.
    entry_manifest = base / cache_key / "manifest.json"

    def _cache_complete() -> bool:
        return entry_manifest.exists()

    if _cache_complete():
        _logger.debug(
            lambda: f"Cache {cache_key} already populated; skipping lock acquire."
        )
        yield
        return

    # ``thread_local=False`` is required: the acquire runs on an
    # ``asyncio.to_thread`` worker, but the release fires from the finally
    # below on whatever worker the event loop picks next. With the default
    # (thread-local counter) the release runs on a thread whose TLS doesn't
    # know about the acquire and silently no-ops, leaving the OS lock held
    # forever.
    lock: FileLock | SoftFileLock = FileLock(
        str(lock_path), mode=_LOCK_FILE_MODE, thread_local=False
    )
    try:
        try:
            acquired = await asyncio.to_thread(
                _blocking_acquire, lock, timeout, lock_path, _cache_complete
            )
        except NotImplementedError as e:
            if _FLOCK_UNSUPPORTED_HINT not in str(e):
                raise
            _logger.warning(
                lambda: (
                    f"Filesystem at {lock_path} does not support flock; "
                    f"falling back to SoftFileLock (less robust on crash)."
                )
            )
            lock = SoftFileLock(
                str(lock_path), mode=_LOCK_FILE_MODE, thread_local=False
            )
            acquired = await asyncio.to_thread(
                _blocking_acquire, lock, timeout, lock_path, _cache_complete
            )
    except Timeout:
        # A populator SIGKILLed *before* completing (e.g. on the SoftFileLock/NFS
        # path, where the lock tombstone persists) leaves an incomplete entry, so
        # the cache-complete bypass never fires and every waiter blocks the full
        # timeout. Rather than fail the whole run, degrade to an unlocked
        # populate: mmap_cache.populate writes to a tmp dir and atomically
        # renames it into place, so a concurrent unlocked populate is safe --
        # it may duplicate the tokenize work but cannot corrupt the entry.
        _logger.warning(
            lambda: (
                f"Timed out acquiring mmap-cache lock at {lock_path} after "
                f"{timeout}s; proceeding with an unlocked populate (populate is "
                f"atomic, so this is safe but may repeat the tokenize)."
            )
        )
        acquired = False
    # SoftFileLock only applies mode via os.open (umask-masked); FileLock
    # fchmods. Also covers filelock>=3.24, which auto-mutates FileLock to
    # SoftFileLock on flock ENOSYS without raising NotImplementedError.
    # Harmless when FileLock already forced 0o664.
    if acquired:
        await asyncio.to_thread(_ensure_lock_file_mode, lock_path)
    try:
        yield
    finally:
        if acquired:
            try:
                await asyncio.to_thread(lock.release)
            except OSError:
                _logger.debug(
                    lambda: (
                        f"Best-effort release of mmap-cache lock {lock_path} failed."
                    )
                )
