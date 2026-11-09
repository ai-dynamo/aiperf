# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real service-process ancestry, queue transport, and controller-death regression."""

import asyncio
import os
import signal
import sys
import threading
from contextlib import suppress
from pathlib import Path

import psutil
import pytest
from pytest import param

from aiperf.common.constants import IS_LINUX

# Run outside pytest so its clock/singleton patches cannot cross process boundaries.
# Replace only the service workload; exercise the real manager and death guard.
_CONTROLLER = """
import asyncio
import multiprocessing as mp
import os
import sys
from queue import Empty

from aiperf.common.bootstrap import _install_parent_death_signal
from aiperf.controller import multiprocess_service_manager as services
from aiperf.plugin.enums import ServiceType


def guarded_service(*, controller_pid: int, log_queue: mp.Queue, **kwargs) -> None:
    _install_parent_death_signal(controller_pid)
    log_queue.put((os.getppid(), mp.get_start_method()))
    asyncio.run(wait_forever())


async def wait_forever() -> None:
    await asyncio.Event().wait()


async def main() -> None:
    method = sys.argv[1]
    mp.set_start_method(method)
    queue = mp.Queue()
    services.bootstrap_and_run_service = guarded_service
    manager = services.MultiProcessServiceManager(
        required_services={}, run=None, log_queue=queue,
    )
    try:
        await manager.run_service(ServiceType.DATASET_MANAGER)
        child = manager.multi_process_info[0].process
        print(child.pid, flush=True)
        try:
            parent_pid, child_method = await asyncio.to_thread(queue.get, True, 20)
        except Empty:
            raise AssertionError(f"Service failed to start: exitcode={child.exitcode}") from None
        assert parent_pid == os.getpid()
        assert child_method == ("spawn" if method == "forkserver" else method)
        assert mp.get_start_method() == method
        assert child.is_alive()
        print("ready", flush=True)
        await wait_forever()
    finally:
        for info in manager.multi_process_info:
            if info.process.is_alive():
                info.process.kill()
            info.process.join(timeout=5)
        queue.close()
        queue.join_thread()


if __name__ == "__main__":
    asyncio.run(main())
"""


@pytest.mark.skipif(
    not IS_LINUX,
    reason="PR_SET_PDEATHSIG is Linux-only",
)
@pytest.mark.parametrize(
    "method",
    [
        param("forkserver", id="forkserver"),
        param("spawn", id="spawn"),
        param("fork", id="fork"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_service_startup_and_controller_death(
    method: str, tmp_path: Path
) -> None:
    """A ready service exchanges queue data and dies on controller SIGKILL."""
    script = tmp_path / "controller.py"
    script.write_text(_CONTROLLER)
    controller = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        method,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        pid_line = await asyncio.wait_for(controller.stdout.readline(), timeout=30)
        assert pid_line.strip().isdigit(), await controller.stderr.read()
        child = psutil.Process(int(pid_line))
        ready = await asyncio.wait_for(controller.stdout.readline(), timeout=30)
        assert ready == b"ready\n", await controller.stderr.read()
        controller.kill()
        await asyncio.wait_for(controller.wait(), timeout=10)
        for _ in range(100):
            try:
                if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            await asyncio.to_thread(threading.Event().wait, 0.1)
        else:
            pytest.fail("service survived controller SIGKILL")
    finally:
        # Also clean the forkserver/resource tracker, or a child whose guard failed.
        with suppress(ProcessLookupError):
            os.killpg(controller.pid, signal.SIGKILL)
        await asyncio.wait_for(controller.communicate(), timeout=10)
