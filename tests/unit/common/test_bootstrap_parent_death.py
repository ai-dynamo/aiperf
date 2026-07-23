# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The parent-death guard installs a kernel-level PR_SET_PDEATHSIG(SIGKILL) backstop so spawned services die when their controller does."""

import os
import platform
import subprocess
import sys
import time
from unittest import mock

import pytest
from pytest import param

from aiperf.common.bootstrap import _install_parent_death_signal

IS_LINUX = platform.system() == "Linux"
PR_SET_PDEATHSIG = 1


def test_install_parent_death_signal_arms_sigkill_on_linux():
    """On Linux it must call prctl(PR_SET_PDEATHSIG, SIGKILL)."""
    import signal

    fake_libc = mock.Mock()
    fake_libc.prctl.return_value = 0
    with (
        mock.patch("aiperf.common.bootstrap.IS_LINUX", True),
        mock.patch("ctypes.CDLL", return_value=fake_libc),
        mock.patch.object(os, "getppid", return_value=4242),
    ):
        _install_parent_death_signal(controller_pid=4242)

    fake_libc.prctl.assert_called_once_with(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)


def test_install_parent_death_signal_noop_on_non_linux():
    """On non-Linux platforms it must not touch ctypes/prctl at all."""
    with (
        mock.patch("aiperf.common.bootstrap.IS_LINUX", False),
        mock.patch("ctypes.CDLL") as cdll,
    ):
        _install_parent_death_signal(controller_pid=4242)

    cdll.assert_not_called()


def test_install_parent_death_signal_exits_if_controller_already_died():
    """If our parent is no longer the controller, the controller died before the guard armed, so the child must exit itself."""
    fake_libc = mock.Mock()
    fake_libc.prctl.return_value = 0
    with (
        mock.patch("aiperf.common.bootstrap.IS_LINUX", True),
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
        mock.patch("aiperf.common.bootstrap.IS_LINUX", True),
        mock.patch("ctypes.CDLL", return_value=fake_libc),
        mock.patch.object(os, "getppid", return_value=4242),
        mock.patch.object(os, "_exit", side_effect=SystemExit) as exit_mock,
    ):
        _install_parent_death_signal(controller_pid=4242)

    exit_mock.assert_not_called()


def test_install_parent_death_signal_falls_back_to_getppid_snapshot():
    """With no controller_pid it snapshots getppid() and does not exit when that parent is stable."""
    fake_libc = mock.Mock()
    fake_libc.prctl.return_value = 0
    with (
        mock.patch("aiperf.common.bootstrap.IS_LINUX", True),
        mock.patch("ctypes.CDLL", return_value=fake_libc),
        mock.patch.object(os, "getppid", return_value=999),
        mock.patch.object(os, "_exit", side_effect=SystemExit) as exit_mock,
    ):
        _install_parent_death_signal()

    exit_mock.assert_not_called()


def test_install_parent_death_signal_prctl_failure_returns_early():
    """A nonzero prctl return means the guard never armed, so the reparent check is skipped and the process must not exit."""
    fake_libc = mock.Mock()
    fake_libc.prctl.return_value = -1
    with (
        mock.patch("aiperf.common.bootstrap.IS_LINUX", True),
        mock.patch("ctypes.CDLL", return_value=fake_libc),
        # A mismatched ppid would trigger the race path if it were reached.
        mock.patch.object(os, "getppid", return_value=999),
        mock.patch.object(os, "_exit", side_effect=SystemExit) as exit_mock,
    ):
        _install_parent_death_signal(controller_pid=1234)

    exit_mock.assert_not_called()


@pytest.mark.parametrize(
    "exc_type",
    [
        param(OSError, id="cdll_load_failure"),
        param(AttributeError, id="prctl_symbol_missing"),
    ],
)  # fmt: skip
def test_install_parent_death_signal_libc_failure_returns_early(
    exc_type: type[Exception],
):
    """libc load / prctl symbol failures are non-fatal best-effort: no crash, no exit."""
    with (
        mock.patch("aiperf.common.bootstrap.IS_LINUX", True),
        mock.patch("ctypes.CDLL", side_effect=exc_type("boom")),
        mock.patch.object(os, "getppid", return_value=999),
        mock.patch.object(os, "_exit", side_effect=SystemExit) as exit_mock,
    ):
        _install_parent_death_signal(controller_pid=1234)

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
    """End-to-end: SIGKILL the parent and the armed grandchild must die on its own via real kernel PR_SET_PDEATHSIG delivery."""
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
