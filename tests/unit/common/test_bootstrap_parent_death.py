# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the parent-death guard installed in spawned service processes.

The SystemController launches each service as a ``daemon=True`` child (via
``fork`` on Linux, ``spawn`` on macOS+dashboard). ``daemon`` only reaps children
when the parent exits *cleanly* through Python's ``atexit`` hook. When the
controller is SIGKILL'd (agent timeout, OOM, hard kill), that hook never runs
and the services orphan, reparenting to init/systemd and leaking RAM
indefinitely. ``_install_parent_death_signal`` installs a kernel-level
``PR_SET_PDEATHSIG(SIGKILL)`` backstop so the kernel reaps each service the
instant its controller dies, however it dies.
"""

import os
import platform
import subprocess
import sys
import time
from unittest import mock

import pytest

from aiperf.common.bootstrap import _install_parent_death_signal

IS_LINUX = platform.system() == "Linux"
PR_SET_PDEATHSIG = 1


def test_install_parent_death_signal_arms_sigkill_on_linux():
    """On Linux it must call prctl(PR_SET_PDEATHSIG, SIGKILL)."""
    import signal

    fake_libc = mock.Mock()
    fake_libc.prctl.return_value = 0
    with (
        mock.patch.object(platform, "system", return_value="Linux"),
        mock.patch("ctypes.CDLL", return_value=fake_libc),
        mock.patch.object(os, "getppid", return_value=4242),
    ):
        _install_parent_death_signal(controller_pid=4242)

    fake_libc.prctl.assert_called_once_with(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)


def test_install_parent_death_signal_noop_on_non_linux():
    """On non-Linux platforms it must not touch ctypes/prctl at all."""
    with (
        mock.patch.object(platform, "system", return_value="Darwin"),
        mock.patch("ctypes.CDLL") as cdll,
    ):
        _install_parent_death_signal(controller_pid=4242)

    cdll.assert_not_called()


def test_install_parent_death_signal_exits_if_controller_already_died():
    """If our live parent is no longer the controller, the controller died in
    the launch/import window before the guard armed, so the death signal was
    missed forever — the child must exit itself.
    """
    fake_libc = mock.Mock()
    fake_libc.prctl.return_value = 0
    with (
        mock.patch.object(platform, "system", return_value="Linux"),
        mock.patch("ctypes.CDLL", return_value=fake_libc),
        # Controller was 4242, but we have reparented to a subreaper (1).
        mock.patch.object(os, "getppid", return_value=1),
        mock.patch.object(os, "_exit", side_effect=SystemExit) as exit_mock,
        pytest.raises(SystemExit),
    ):
        _install_parent_death_signal(controller_pid=4242)

    exit_mock.assert_called_once()


def test_install_parent_death_signal_no_exit_when_controller_alive():
    """When our parent is still the controller, it must not exit."""
    fake_libc = mock.Mock()
    fake_libc.prctl.return_value = 0
    with (
        mock.patch.object(platform, "system", return_value="Linux"),
        mock.patch("ctypes.CDLL", return_value=fake_libc),
        mock.patch.object(os, "getppid", return_value=4242),
        mock.patch.object(os, "_exit", side_effect=SystemExit) as exit_mock,
    ):
        _install_parent_death_signal(controller_pid=4242)

    exit_mock.assert_not_called()


def test_install_parent_death_signal_falls_back_to_getppid_snapshot():
    """With no controller_pid (e.g. tests), it snapshots getppid() and does not
    exit when that parent is stable — preserving correctness under fork."""
    fake_libc = mock.Mock()
    fake_libc.prctl.return_value = 0
    with (
        mock.patch.object(platform, "system", return_value="Linux"),
        mock.patch("ctypes.CDLL", return_value=fake_libc),
        mock.patch.object(os, "getppid", return_value=999),
        mock.patch.object(os, "_exit", side_effect=SystemExit) as exit_mock,
    ):
        _install_parent_death_signal()

    exit_mock.assert_not_called()


# Program run as both "parent" and "child" by the real-kill integration test.
# parent: spawns a child, passing its OWN pid as the controller_pid, prints the
#         child pid, then sleeps.
# child:  arms the guard against that controller_pid, then sleeps.
# Killing the parent must make the child die — either via PR_SET_PDEATHSIG (if
# armed before the parent died) or via the getppid()-mismatch self-exit (if the
# parent died during the child's import/launch window).
_REAL_KILL_PROG = """
import sys, time, subprocess, os
from aiperf.common.bootstrap import _install_parent_death_signal

role = sys.argv[1]
if role == "child":
    _install_parent_death_signal(controller_pid=int(sys.argv[2]))
    print("armed", flush=True)
    time.sleep(120)
else:
    child = subprocess.Popen(
        [sys.executable, "-c", sys.argv[2], "child", str(os.getpid())]
    )
    print(child.pid, flush=True)
    time.sleep(120)
"""


def _pid_alive(pid: int) -> bool:
    """True if pid exists and is not a zombie (reaped-but-not-cleaned)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Distinguish a live process from a zombie awaiting reap by its parent.
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().split(") ", 1)[1][0]
        return state != "Z"
    except (FileNotFoundError, IndexError):
        return False


@pytest.mark.skipif(not IS_LINUX, reason="PR_SET_PDEATHSIG is Linux-only")
def test_parent_death_signal_real_kill_reaps_child():
    """End-to-end: SIGKILL the parent, the armed grandchild must die on its own.

    This is the real proof of the fix — it exercises actual kernel
    PR_SET_PDEATHSIG delivery (and the getppid mismatch fallback), not a mock.
    """
    parent = subprocess.Popen(
        [sys.executable, "-c", _REAL_KILL_PROG, "parent", _REAL_KILL_PROG],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        # First line printed by the parent is the grandchild pid.
        child_pid = int(parent.stdout.readline().strip())
        assert _pid_alive(child_pid), "grandchild should be alive before kill"

        # Hard-kill the parent (simulates a SIGKILL'd SystemController).
        parent.kill()
        parent.wait(timeout=10)

        # The grandchild must die without anyone signalling it directly.
        deadline = time.time() + 10
        while time.time() < deadline:
            if not _pid_alive(child_pid):
                break
            time.sleep(0.05)
        assert not _pid_alive(child_pid), (
            f"grandchild {child_pid} survived parent death (parent-death guard "
            "did not fire)"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
