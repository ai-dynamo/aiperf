# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import select
import signal
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiperf.common.constants import IS_WINDOWS

if IS_WINDOWS:
    pytest.skip("LinuxDriver is POSIX-only", allow_module_level=True)

from textual import events

from aiperf.ui.dashboard.aiperf_dashboard_ui import AIPerfDashboardUI
from aiperf.ui.dashboard.lenient_linux_driver import LenientLinuxDriver

_JOIN_TIMEOUT = 5.0


@pytest.fixture(autouse=True)
def _restore_signal_handlers() -> Iterator[None]:
    # LinuxDriver.__init__ installs process-wide SIGTSTP/SIGCONT handlers.
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTSTP, signal.SIGCONT)}
    yield
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


@pytest.fixture
def stdin_pipe(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[int, int]]:
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(
        sys,
        "__stdin__",
        SimpleNamespace(fileno=lambda: read_fd, isatty=lambda: False),
    )
    yield read_fd, write_fd
    for fd in (read_fd, write_fd):
        with suppress(OSError):
            os.close(fd)


def _wait_until_drained(read_fd: int) -> None:
    deadline = time.monotonic() + _JOIN_TIMEOUT
    while select.select([read_fd], [], [], 0.01)[0]:
        if time.monotonic() > deadline:
            pytest.fail("driver never drained stdin")


@pytest.mark.asyncio
async def test_lenient_driver_run_input_thread_undecodable_byte_keeps_reading(
    stdin_pipe: tuple[int, int],
) -> None:
    read_fd, write_fd = stdin_pipe
    driver = LenientLinuxDriver(MagicMock())
    key_seen = threading.Event()
    received: list[events.Message] = []

    def record(message: events.Message) -> None:
        received.append(message)
        if isinstance(message, events.Key) and message.key == "a":
            key_seen.set()

    driver.process_message = record  # type: ignore[method-assign]
    thread = threading.Thread(target=driver.run_input_thread, daemon=True)
    try:
        thread.start()
        os.write(write_fd, b"\xff")
        _wait_until_drained(read_fd)
        os.write(write_fd, b"a")
        assert key_seen.wait(_JOIN_TIMEOUT), received
    finally:
        driver.exit_event.set()
        thread.join(_JOIN_TIMEOUT)
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_dashboard_ui_app_uses_lenient_driver_on_posix() -> None:
    run = MagicMock()
    run.cfg.comm_config.comm_backend = "zmq_ipc"
    ui = AIPerfDashboardUI(
        log_queue=MagicMock(),
        run=run,
        controller=MagicMock(),
    )

    assert ui.app.driver_class is LenientLinuxDriver
