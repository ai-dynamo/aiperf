# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A child guarded by the parent-death watchdog must stop holding inherited pipes once its parent dies."""

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from aiperf.common.constants import IS_WINDOWS
from aiperf.common.parent_death_watchdog import start_parent_death_watchdog

_THREAD_NAME = "parent-death-watchdog"


def _watchdog_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == _THREAD_NAME]


@pytest.fixture
def exit_codes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record ``os._exit`` calls instead of ending the test process."""
    codes: list[int] = []
    monkeypatch.setattr("aiperf.common.parent_death_watchdog.os._exit", codes.append)
    return codes


def test_start_parent_death_watchdog_without_parent_starts_no_thread(
    monkeypatch: pytest.MonkeyPatch, exit_codes: list[int]
) -> None:
    monkeypatch.setattr(
        "aiperf.common.parent_death_watchdog.multiprocessing.parent_process",
        lambda: None,
    )
    before = len(_watchdog_threads())

    finished = start_parent_death_watchdog("Probe", 0.0, logging.getLogger("t"))

    assert isinstance(finished, threading.Event)
    assert not finished.is_set()
    assert len(_watchdog_threads()) == before
    assert exit_codes == []


def test_start_parent_death_watchdog_dead_parent_past_deadline_exits_with_1(
    monkeypatch: pytest.MonkeyPatch,
    exit_codes: list[int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _DeadParent:
        def join(self, timeout: float | None = None) -> None:
            return None

    monkeypatch.setattr(
        "aiperf.common.parent_death_watchdog.multiprocessing.parent_process",
        lambda: _DeadParent(),
    )

    with caplog.at_level(logging.ERROR):
        start_parent_death_watchdog("Probe", 0.0, logging.getLogger("t"))
        for thread in _watchdog_threads():
            thread.join(timeout=5.0)

    assert exit_codes == [1]
    assert "Probe cleanup exceeded 0s after parent death" in caplog.text


def test_start_parent_death_watchdog_cleanup_finished_in_time_does_not_exit(
    monkeypatch: pytest.MonkeyPatch, exit_codes: list[int]
) -> None:
    parent_died = threading.Event()

    class _ParentDyingOnSignal:
        def join(self, timeout: float | None = None) -> None:
            parent_died.wait()

    monkeypatch.setattr(
        "aiperf.common.parent_death_watchdog.multiprocessing.parent_process",
        lambda: _ParentDyingOnSignal(),
    )

    finished = start_parent_death_watchdog("Probe", 30.0, logging.getLogger("t"))
    finished.set()
    parent_died.set()
    for thread in _watchdog_threads():
        thread.join(timeout=5.0)

    assert exit_codes == []


# Parent spawns a guarded child that inherits stdout, then SIGKILLs itself.
# The test reads that pipe: EOF arrives only once every writer is gone, which
# is exactly what a caller running ``aiperf ... | tee`` waits on.
_PIPE_HOLDER_PROG = """
import logging, multiprocessing as mp, os, signal, sys, time


def child(ready, deadline):
    from aiperf.common.parent_death_watchdog import start_parent_death_watchdog

    start_parent_death_watchdog("Probe", deadline, logging.getLogger("probe"))
    ready.set()
    time.sleep(120)  # stands in for an HTTP call that never returns


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=child, args=(ready, float(sys.argv[1])), daemon=True)
    proc.start()
    if not ready.wait(60):
        sys.exit("child never armed the watchdog")
    print(proc.pid, flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
"""


@pytest.mark.skipif(IS_WINDOWS, reason="signal.SIGKILL is unavailable on Windows")
def test_parent_death_watchdog_releases_inherited_pipe_after_parent_sigkill(
    tmp_path: Path,
) -> None:
    prog = tmp_path / "pipe_holder.py"
    prog.write_text(_PIPE_HOLDER_PROG)
    parent = subprocess.Popen(
        [sys.executable, str(prog), "0.5"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    child_pid = None
    try:
        child_pid = int(parent.stdout.readline().strip())
        rest: list[str] = []
        reader = threading.Thread(target=lambda: rest.append(parent.stdout.read()))
        reader.start()
        reader.join(timeout=20.0)

        assert not reader.is_alive(), (
            "pipe never reached EOF: the child kept the dead parent's stdout open"
        )
        assert "Probe cleanup exceeded" in rest[0]
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=10)
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
