# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards the daemon-fork path for LCB code-execution grading.

AIPerf spawns every service as a daemon process
(``multiprocess_service_manager.py``: ``daemon=True``). The LCB
``code_execution`` grader runs lighteval's ``codegen_metrics``, which fans out
to a ``ProcessPoolExecutor`` — and Python forbids daemon processes from
spawning children. Unit tests mock ``codegen_metrics``, so they never hit the
fork restriction; that gap let LCB grading ship 100%-``unparsed`` (the daemon
error was caught and mislabeled).

These tests close that gap by exercising the real thing: spawning a
``ProcessPoolExecutor`` from inside a genuine daemon process. The negative case
pins *why* the workaround is needed (without it the spawn raises); the positive
case proves ``allow_daemon_children`` lets grading fan out.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import pytest

pytestmark = pytest.mark.component_integration


def _double(x: int) -> int:
    return x * 2


def _pool_without_guard(queue: mp.Queue) -> None:
    """Run a ProcessPoolExecutor with no daemon workaround (mirrors the
    pre-fix grader). Reports the outcome back over ``queue``."""
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            list(executor.map(_double, [1, 2, 3]))
        queue.put(("ok", None))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", repr(exc)))


def _pool_with_guard(queue: mp.Queue) -> None:
    """Run the same fan-out under ``allow_daemon_children`` (mirrors the fix)."""
    from aiperf.common.utils import allow_daemon_children

    try:
        with allow_daemon_children(), ProcessPoolExecutor(max_workers=1) as executor:
            result = list(executor.map(_double, [1, 2, 3]))
        queue.put(("ok", result))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", repr(exc)))


def _run_in_daemon(target) -> tuple[str, object]:
    """Run ``target(queue)`` inside a real daemon process and return its report.

    Uses the default start method so this matches how AIPerf actually spawns
    services on each platform (spawn on macOS/Windows, fork on Linux).
    """
    ctx = mp.get_context()
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(queue,), daemon=True)
    proc.start()
    try:
        status, payload = queue.get(timeout=120)
    finally:
        proc.join(timeout=30)
    return status, payload


@pytest.mark.slow
class TestDaemonProcessPoolSpawn:
    def test_daemon_cannot_spawn_pool_without_guard(self) -> None:
        """Pins the restriction the fix exists for: a daemon process spawning a
        ProcessPoolExecutor raises 'daemonic processes are not allowed to have
        children'. This is the failure the LCB grader hit."""
        status, payload = _run_in_daemon(_pool_without_guard)
        assert status == "error", f"expected daemon spawn to fail, got: {payload}"
        assert "daemonic processes are not allowed to have children" in str(payload)

    def test_daemon_can_spawn_pool_with_guard(self) -> None:
        """``allow_daemon_children`` lets the same daemon process fan out — the
        exact path LCB grading relies on."""
        status, payload = _run_in_daemon(_pool_with_guard)
        assert status == "ok", f"expected daemon spawn to succeed, got: {payload}"
        assert payload == [2, 4, 6]
