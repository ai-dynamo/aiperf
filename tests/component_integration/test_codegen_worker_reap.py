# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-process guard for the codegen worker's process-group teardown.

When a grade times out, ``CodegenGradingWorker`` must tear down the worker AND
lighteval's forked sandbox grandchildren (``start_new_session`` + ``killpg``),
not leave them running. This uses real subprocesses and real time, so it lives
in component_integration (the unit suite patches ``asyncio.sleep`` to run
instantly, which would defeat the reaping poll)."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path

import pytest

from aiperf.accuracy.graders._codegen_worker_client import (
    CodegenGradingWorker,
    CodegenWorkerError,
)

pytestmark = pytest.mark.component_integration


@pytest.mark.skipif(
    not hasattr(os, "killpg"), reason="process groups unavailable on this platform"
)
@pytest.mark.asyncio
async def test_timeout_kill_reaps_sandbox_grandchildren(tmp_path: Path) -> None:
    pidfile = tmp_path / "gc.pid"
    script = tmp_path / "reap_worker.py"
    # The worker spawns a long-lived grandchild (stand-in for lighteval's forked
    # sandbox) then hangs so the client times out and kills it. The grandchild
    # detaches its std fds so it doesn't hold the worker's pipes open.
    script.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys, time
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(300)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open({str(pidfile)!r}, "w") as f:
                f.write(str(child.pid))
                f.flush()
            time.sleep(300)
            """
        )
    )
    worker = CodegenGradingWorker(worker_cmd=[sys.executable, str(script)])
    try:
        with pytest.raises(CodegenWorkerError):
            await worker.grade_codegen([{"input_output": "{}"}], [["x"]], timeout=1.0)

        for _ in range(100):  # wait for the grandchild pid to be recorded
            if pidfile.exists() and pidfile.read_text().strip():
                break
            await asyncio.sleep(0.05)
        gc_pid = int(pidfile.read_text().strip())

        reaped = False
        for _ in range(100):  # up to ~5s for the group-kill to reap it
            try:
                os.kill(gc_pid, 0)
            except ProcessLookupError:
                reaped = True
                break
            await asyncio.sleep(0.05)
        assert reaped, f"sandbox grandchild {gc_pid} still alive after worker kill"
    finally:
        await worker.aclose()


@pytest.mark.skipif(
    not hasattr(os, "killpg"), reason="process groups unavailable on this platform"
)
@pytest.mark.asyncio
async def test_death_watcher_reaps_session_when_parent_dies(tmp_path: Path) -> None:
    # Abrupt-exit path: the real _start_death_watcher must reap the worker's whole
    # session (worker + forked sandbox children) when the parent's death-pipe
    # write end closes, even while the worker is busy and not reading stdin.
    pidfile = tmp_path / "gc.pid"
    script = tmp_path / "death_probe.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os, subprocess, sys, time
            os.setsid()  # own session so killpg(0) can't touch the test runner
            r, w = os.pipe()
            from aiperf.accuracy.graders import _codegen_worker as wkr
            os.environ[wkr._DEATH_FD_ENV] = str(r)
            wkr._start_death_watcher()
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(300)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open({str(pidfile)!r}, "w") as f:
                f.write(str(child.pid))
                f.flush()
            os.close(w)  # simulate the parent's exit -> EOF -> watcher killpg(0)
            time.sleep(300)  # busy; wait to be killed
            """
        )
    )
    proc = await asyncio.create_subprocess_exec(sys.executable, str(script))
    rc = await asyncio.wait_for(proc.wait(), timeout=30)
    assert rc < 0, f"probe should have been signal-killed, got rc={rc}"

    gc_pid = int(pidfile.read_text().strip())
    reaped = False
    for _ in range(100):  # up to ~5s for the group-kill to reap the grandchild
        try:
            os.kill(gc_pid, 0)
        except ProcessLookupError:
            reaped = True
            break
        await asyncio.sleep(0.05)
    assert reaped, f"sandbox grandchild {gc_pid} survived the parent's death"
