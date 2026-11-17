# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound how long a multiprocessing child outlives its parent.

A child inherits its parent's stdout/stderr. When the parent dies without
cleaning the child up (SIGKILL, OOM, crash), a child blocked in a network call
keeps that pipe open, so a caller capturing output (``aiperf ... | tee``) never
sees EOF.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time

# Patched by tests instead of ``os._exit`` itself, which is shared by every
# thread in the test process.
_exit = os._exit


def start_parent_death_watchdog(
    name: str, deadline_seconds: float, logger: logging.Logger
) -> None:
    """Force-exit this process ``deadline_seconds`` after its parent dies.

    Waits on the parent's sentinel in a daemon thread, which works under fork,
    spawn and forkserver and on every platform (``PR_SET_PDEATHSIG`` is
    Linux-only and fires when the creating *thread* exits). Once the parent is
    gone, the process gets ``deadline_seconds`` to finish whatever it is doing,
    including interpreter teardown, then ``os._exit(1)``s so the inherited
    pipes close.

    There is deliberately no way to disarm it: while the parent lives the thread
    only waits and dies with the process, and after the parent dies a disarm
    could only exempt the exit path (thread joins, queue feeder joins, SDK
    atexit hooks), which can block on the network as long as the work itself.

    Arm it before any call that can block. When there is no multiprocessing
    parent (run in-process), no thread is started.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def _watch() -> None:
        parent.join()
        time.sleep(deadline_seconds)
        logger.error(
            "%s cleanup exceeded %.0fs after parent death; exiting now.",
            name,
            deadline_seconds,
        )
        _exit(1)

    threading.Thread(target=_watch, name="parent-death-watchdog", daemon=True).start()
