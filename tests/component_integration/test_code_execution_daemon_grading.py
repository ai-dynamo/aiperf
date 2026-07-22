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
    except Exception as exc:
        queue.put(("error", repr(exc)))


def _pool_with_guard(queue: mp.Queue) -> None:
    """Run the same fan-out under ``allow_daemon_children`` (mirrors the fix)."""
    from aiperf.common.utils import allow_daemon_children

    try:
        with allow_daemon_children(), ProcessPoolExecutor(max_workers=1) as executor:
            result = list(executor.map(_double, [1, 2, 3]))
        queue.put(("ok", result))
    except Exception as exc:
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


def _nested_target_process(queue: mp.Queue) -> None:
    """Reproduce lighteval's check_correctness mechanism: a NESTED local function
    passed as the multiprocessing.Process target, with no fork guard. Under
    spawn/forkserver the target can't be pickled, so the child never runs and the
    "result" stays empty — exactly how LCB grading scored every test as failed."""

    def _inner(result):  # nested/local, matches lighteval's ``_temp_run``
        result.append("ran")

    try:
        manager = mp.Manager()
        result = manager.list()
        p = mp.Process(target=_inner, args=(result,))
        p.start()
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
        # Empty result == the child never executed (the bug).
        queue.put(("ok", list(result)))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def _nested_target_process_with_fork(queue: mp.Queue) -> None:
    """Same nested-target mechanism, but under ``use_fork_start_method`` (the fix):
    fork inherits the parent's memory so the nested target needs no pickling and
    the child actually runs."""
    from aiperf.common.utils import allow_daemon_children, use_fork_start_method

    def _inner(result):
        result.append("ran")

    try:
        with allow_daemon_children(), use_fork_start_method():
            manager = mp.Manager()
            result = manager.list()
            p = mp.Process(target=_inner, args=(result,))
            p.start()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
        queue.put(("ok", list(result)))
    except Exception as exc:
        queue.put(("error", repr(exc)))


def _run_in_process(target, start_method: str) -> tuple[str, object]:
    """Run ``target(queue)`` in a child using an explicit start method."""
    ctx = mp.get_context(start_method)
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(queue,), daemon=True)
    proc.start()
    try:
        status, payload = queue.get(timeout=120)
    finally:
        proc.join(timeout=30)
    return status, payload


@pytest.mark.slow
@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="fork start method unavailable on this platform",
)
class TestForkStartMethodForCodegen:
    """Guards issue #1145: lighteval's LCB codegen sandbox passes a nested local
    function as the Process target, which only works under fork. These tests
    exercise that real mechanism (no lighteval dependency) under spawn."""

    def test_nested_target_never_runs_under_spawn_without_guard(self) -> None:
        """Pins the bug: under spawn, a nested-local Process target can't be
        pickled, the child never runs, and the result comes back empty — which is
        exactly how check_correctness scored every LCB test case as failed."""
        status, payload = _run_in_process(_nested_target_process, "spawn")
        # Either the result is empty (child never ran) or the spawn raised; both
        # demonstrate the mechanism. The key point is the target did NOT run.
        assert status == "error" or payload == [], (
            f"expected nested target to fail under spawn, got: {status} {payload}"
        )

    def test_nested_target_runs_under_spawn_with_fork_guard(self) -> None:
        """``use_fork_start_method`` makes the same nested-target Process actually
        run even when the parent default is spawn — the fix for #1145."""
        status, payload = _run_in_process(_nested_target_process_with_fork, "spawn")
        assert status == "ok", f"expected fork-guarded run to succeed, got: {payload}"
        assert payload == ["ran"]
