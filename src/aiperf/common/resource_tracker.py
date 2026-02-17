# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for unregistering multiprocessing semaphores from the resource tracker.

When a process exits via os._exit(), garbage collection is skipped, so semaphores
created by multiprocessing.Lock/RLock/Queue are never properly released. The
resource tracker then warns about "leaked semaphore objects". These helpers
explicitly unregister semaphores before os._exit() to suppress those warnings.
"""

import contextlib
from multiprocessing import resource_tracker


def unregister_semaphore(lock: object) -> None:
    """Unregister a single lock/semaphore object from the resource tracker.

    Works with any object that has a `_semlock` attribute (multiprocessing.Lock,
    RLock, BoundedSemaphore, etc.). No-op if the lock has no semaphore.
    """
    sem = getattr(lock, "_semlock", None)
    if sem is not None:
        with contextlib.suppress(Exception):
            resource_tracker.unregister(sem.name, "semaphore")


def unregister_queue_semaphores(q: object) -> None:
    """Unregister a multiprocessing.Queue's internal semaphores from the resource tracker.

    A multiprocessing.Queue creates 3 semaphores (_rlock, _wlock, _sem) that are
    tracked by the resource tracker. Normally these are unregistered when the
    Lock/BoundedSemaphore objects are garbage-collected, but `os._exit()` skips GC.
    """
    for attr in ("_rlock", "_wlock", "_sem"):
        lock = getattr(q, attr, None)
        if lock is not None:
            unregister_semaphore(lock)
