# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The child parent-death guard must survive a non-fork start method.

PR_SET_PDEATHSIG can only cover the *direct* parent. Under forkserver or
spawn that parent is the multiprocessing helper, not the SystemController, so
the guard's old getppid()-equality test was false for every healthy child --
it would have os._exit(1)'d all of them at startup. Liveness of the recorded
controller PID is the question actually being asked.
"""

from __future__ import annotations

import os

import pytest

from aiperf.common import bootstrap


class TestPidIsAlive:
    def test_self_is_alive(self) -> None:
        assert bootstrap._pid_is_alive(os.getpid()) is True

    def test_missing_pid_is_not_alive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(pid: int, sig: int) -> None:
            raise ProcessLookupError

        monkeypatch.setattr(os, "kill", _raise)
        assert bootstrap._pid_is_alive(4242) is False

    def test_permission_denied_counts_as_alive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PID we may not signal still exists -- do not kill ourselves over it."""

        def _raise(pid: int, sig: int) -> None:
            raise PermissionError

        monkeypatch.setattr(os, "kill", _raise)
        assert bootstrap._pid_is_alive(1) is True


class TestParentDeathSignalArming:
    @pytest.fixture(autouse=True)
    def _linux_only(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(bootstrap, "IS_LINUX", True)

    def test_live_controller_under_forkserver_starts_a_watchdog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct parent != controller is the normal forkserver case."""
        started: list[int] = []
        monkeypatch.setattr(bootstrap, "_pid_is_alive", lambda pid: True)
        monkeypatch.setattr(os, "getppid", lambda: 999)
        monkeypatch.setattr(
            bootstrap, "_watch_controller_liveness", lambda pid: started.append(pid)
        )

        bootstrap._install_parent_death_signal(1234)

        assert started == [1234]

    def test_live_controller_under_fork_needs_no_watchdog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started: list[int] = []
        monkeypatch.setattr(bootstrap, "_pid_is_alive", lambda pid: True)
        monkeypatch.setattr(os, "getppid", lambda: 1234)
        monkeypatch.setattr(
            bootstrap, "_watch_controller_liveness", lambda pid: started.append(pid)
        )

        bootstrap._install_parent_death_signal(1234)

        assert started == []

    def test_dead_controller_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bootstrap, "_pid_is_alive", lambda pid: False)
        exits: list[int] = []
        monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))
        monkeypatch.setattr(bootstrap, "_watch_controller_liveness", lambda pid: None)

        bootstrap._install_parent_death_signal(1234)

        assert exits == [1]
