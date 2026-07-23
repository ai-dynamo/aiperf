# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the orphaned-service reaping guards in bootstrap.py.

Covers the SIGUSR1 stack-dump handler registration
(:func:`register_sigusr1_faulthandler`) and the Linux ``PR_SET_PDEATHSIG``
parent-death guard (:func:`_install_parent_death_signal`), including the
reparent-race path where the controller died before the guard was armed.
All prctl/libc interactions are mocked so no real kernel state is touched.
"""

from __future__ import annotations

import ctypes
import faulthandler
import os
import signal
from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.common.bootstrap import (
    _install_parent_death_signal,
    register_sigusr1_faulthandler,
)


class TestRegisterSigusr1Faulthandler:
    """Verify the best-effort SIGUSR1 stack-dump handler registration."""

    @pytest.mark.skipif(
        not hasattr(signal, "SIGUSR1"),
        reason="SIGUSR1 not available on this platform",
    )
    def test_register_sigusr1_faulthandler_installs_real_handler(self) -> None:
        """On platforms with SIGUSR1 the handler must actually be registered
        with faulthandler (unregister returns True only for a registered
        signal)."""
        try:
            register_sigusr1_faulthandler()
            assert faulthandler.unregister(signal.SIGUSR1) is True
        finally:
            # Idempotent: returns False if the assert already unregistered it.
            faulthandler.unregister(signal.SIGUSR1)

    def test_register_sigusr1_faulthandler_passes_expected_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The handler must dump ALL threads (hang debugging needs every
        stack) without chaining to a prior handler."""
        monkeypatch.setattr(signal, "SIGUSR1", 10, raising=False)
        mock_register = MagicMock()
        monkeypatch.setattr(faulthandler, "register", mock_register, raising=False)

        register_sigusr1_faulthandler()

        mock_register.assert_called_once_with(10, all_threads=True, chain=False)

    def test_register_sigusr1_faulthandler_noop_without_sigusr1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On platforms without SIGUSR1 (Windows) the function must return
        before touching faulthandler."""
        monkeypatch.delattr(signal, "SIGUSR1", raising=False)
        mock_register = MagicMock()
        monkeypatch.setattr(faulthandler, "register", mock_register, raising=False)

        register_sigusr1_faulthandler()

        mock_register.assert_not_called()

    @pytest.mark.parametrize(
        "exc_type",
        [
            param(ValueError, id="valueerror_stderr_without_fileno"),
            param(RuntimeError, id="runtimeerror"),
            param(AttributeError, id="attributeerror"),
        ],
    )  # fmt: skip
    def test_register_sigusr1_faulthandler_swallows_register_failure(
        self, monkeypatch: pytest.MonkeyPatch, exc_type: type[Exception]
    ) -> None:
        """Registration is best-effort diagnostics: failures (e.g. a test
        harness wrapping stderr in a stream without ``fileno()``) must not
        propagate into service bootstrap."""
        monkeypatch.setattr(signal, "SIGUSR1", 10, raising=False)
        mock_register = MagicMock(side_effect=exc_type("boom"))
        monkeypatch.setattr(faulthandler, "register", mock_register, raising=False)

        register_sigusr1_faulthandler()  # must not raise

        mock_register.assert_called_once()


class TestInstallParentDeathSignal:
    """Verify the PR_SET_PDEATHSIG guard arming and its failure fallbacks."""

    @pytest.fixture
    def mock_exit(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Intercept os._exit so the reparent-race path can't kill pytest."""
        mock = MagicMock()
        monkeypatch.setattr(os, "_exit", mock)
        return mock

    @pytest.fixture
    def mock_libc(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Mock libc with a successful prctl; patches ctypes.CDLL to return it."""
        libc = MagicMock()
        libc.prctl.return_value = 0
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(return_value=libc))
        return libc

    @pytest.fixture
    def linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force the Linux code path and a real SIGKILL value cross-platform."""
        monkeypatch.setattr("aiperf.common.bootstrap.IS_LINUX", True)
        monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)

    def test_install_parent_death_signal_non_linux_never_touches_ctypes(
        self, monkeypatch: pytest.MonkeyPatch, mock_exit: MagicMock
    ) -> None:
        """PR_SET_PDEATHSIG is Linux-only; other platforms must return before
        loading libc."""
        monkeypatch.setattr("aiperf.common.bootstrap.IS_LINUX", False)
        mock_cdll = MagicMock()
        monkeypatch.setattr(ctypes, "CDLL", mock_cdll)

        _install_parent_death_signal(controller_pid=1234)

        mock_cdll.assert_not_called()
        mock_exit.assert_not_called()

    @pytest.mark.usefixtures("linux")
    def test_install_parent_death_signal_arms_guard_when_parent_alive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_libc: MagicMock,
        mock_exit: MagicMock,
    ) -> None:
        """Happy path: prctl succeeds and the live parent IS the controller,
        so the guard arms with SIGKILL and the process keeps running."""
        monkeypatch.setattr(os, "getppid", lambda: 4242)

        _install_parent_death_signal(controller_pid=4242)

        mock_libc.prctl.assert_called_once_with(1, 9, 0, 0, 0)
        mock_exit.assert_not_called()

    @pytest.mark.usefixtures("linux")
    def test_install_parent_death_signal_none_controller_pid_uses_getppid_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_libc: MagicMock,
        mock_exit: MagicMock,
    ) -> None:
        """Without a controller_pid (tests, direct calls) the guard falls back
        to a getppid() snapshot, which trivially matches under fork."""
        monkeypatch.setattr(os, "getppid", lambda: 777)

        _install_parent_death_signal(controller_pid=None)

        mock_libc.prctl.assert_called_once_with(1, 9, 0, 0, 0)
        mock_exit.assert_not_called()

    @pytest.mark.usefixtures("linux")
    def test_install_parent_death_signal_prctl_failure_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_libc: MagicMock,
        mock_exit: MagicMock,
    ) -> None:
        """A nonzero prctl return means the guard never armed: fall back to
        daemon=True behavior WITHOUT running the reparent check (an unarmed
        guard must never exit the process)."""
        mock_libc.prctl.return_value = -1
        # A mismatched ppid would trigger the race path if it were reached.
        monkeypatch.setattr(os, "getppid", lambda: 999)

        _install_parent_death_signal(controller_pid=1234)

        mock_exit.assert_not_called()

    @pytest.mark.parametrize(
        "exc_type",
        [
            param(OSError, id="cdll_load_failure"),
            param(AttributeError, id="prctl_symbol_missing"),
        ],
    )  # fmt: skip
    @pytest.mark.usefixtures("linux")
    def test_install_parent_death_signal_libc_failure_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_exit: MagicMock,
        exc_type: type[Exception],
    ) -> None:
        """libc load / prctl symbol failures are non-fatal best-effort: no
        crash, no exit, plain fallback to daemon=True behavior."""
        monkeypatch.setattr(ctypes, "CDLL", MagicMock(side_effect=exc_type("boom")))
        monkeypatch.setattr(os, "getppid", lambda: 999)

        _install_parent_death_signal(controller_pid=1234)  # must not raise

        mock_exit.assert_not_called()

    @pytest.mark.usefixtures("linux")
    def test_install_parent_death_signal_reparent_race_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_libc: MagicMock,
        mock_exit: MagicMock,
    ) -> None:
        """If the controller died before the guard armed, the child already
        reparented (getppid != controller_pid) and the death signal will never
        fire — the child must exit immediately rather than orphan."""
        monkeypatch.setattr(os, "getppid", lambda: 999)

        _install_parent_death_signal(controller_pid=1234)

        mock_libc.prctl.assert_called_once_with(1, 9, 0, 0, 0)
        mock_exit.assert_called_once_with(1)
