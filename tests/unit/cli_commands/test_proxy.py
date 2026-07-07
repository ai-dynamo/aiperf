# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone ZMQ proxy CLI's stop-signal installation.

Focuses on the POSIX/Windows split: ``loop.add_signal_handler`` raises
NotImplementedError on the Windows ProactorEventLoop, so the helper must
fall back to ``signal.signal`` there.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable

import pytest

from aiperf.cli_commands.proxy import _install_stop_signal_handlers


class TestInstallStopSignalHandlers:
    @pytest.mark.asyncio
    async def test_posix_registers_loop_handlers_for_sigterm_and_sigint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registrations: list[tuple[signal.Signals, Callable[[], None]]] = []
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        # Pin the POSIX branch so this test passes on windows-latest CI too.
        # IS_WINDOWS is imported lazily inside the helper, so patch the source.
        monkeypatch.setattr("aiperf.common.constants.IS_WINDOWS", False)
        monkeypatch.setattr(
            loop,
            "add_signal_handler",
            lambda sig, callback: registrations.append((sig, callback)),
        )

        _install_stop_signal_handlers(loop, stop_event)

        assert [sig for sig, _ in registrations] == [signal.SIGTERM, signal.SIGINT]
        for _, callback in registrations:
            callback()
        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_windows_falls_back_to_signal_signal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_sigbreak = 21  # Windows SIGBREAK value; absent on POSIX.
        signal_registrations: list[tuple[int, Callable[[int, object], None]]] = []
        loop_registrations: list[tuple[object, ...]] = []
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        monkeypatch.setattr("aiperf.common.constants.IS_WINDOWS", True)
        monkeypatch.setattr(
            loop, "add_signal_handler", lambda *args: loop_registrations.append(args)
        )
        monkeypatch.setattr(
            signal,
            "signal",
            lambda sig, handler: signal_registrations.append((sig, handler)),
        )
        monkeypatch.setattr(signal, "SIGBREAK", fake_sigbreak, raising=False)

        _install_stop_signal_handlers(loop, stop_event)

        assert loop_registrations == []
        assert [sig for sig, _ in signal_registrations] == [
            signal.SIGINT,
            fake_sigbreak,
        ]
        for sig, handler in signal_registrations:
            handler(sig, None)
        # The Windows handler marshals onto the loop via call_soon_threadsafe.
        await asyncio.sleep(0)
        assert stop_event.is_set()
