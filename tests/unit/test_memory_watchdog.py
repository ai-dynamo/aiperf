# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification tests for the per-worker RSS memory watchdog.

These tests exercise the watchdog's marker handling and threshold plumbing
by swapping in a recorder for ``_watchdog_kill_action`` rather than letting
the real ``os._exit(137)`` path fire.
"""

from __future__ import annotations

import time

import pytest

from tests import conftest as tests_conftest
from tests.conftest import _WATCHDOG_SUPPORTED, _watchdog_state

pytestmark = pytest.mark.skipif(
    not _WATCHDOG_SUPPORTED,
    reason="memory watchdog requires Linux /proc/self/status",
)


@pytest.mark.memory_limit(mb=64)
def test_watchdog_fires_when_rss_exceeds_threshold(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int, int]] = []

    def recorder(nodeid: str, rss_bytes: int, threshold_bytes: int) -> None:
        calls.append((nodeid, rss_bytes, threshold_bytes))

    monkeypatch.setattr(tests_conftest, "_watchdog_kill_action", recorder)

    # Force RSS well past the 64 MiB threshold. Writing into the buffer
    # ensures the pages are resident, not just reserved.
    buf = bytearray(256 * 1024 * 1024)
    for i in range(0, len(buf), 4096):
        buf[i] = 1

    # Three watchdog intervals (500 ms each) to guarantee a sample fires.
    time.sleep(1.5)

    assert len(calls) == 1, f"expected exactly one kill call, got {calls}"
    nodeid, rss_bytes, threshold_bytes = calls[0]
    assert nodeid == request.node.nodeid
    assert rss_bytes > 64 * 1024 * 1024
    assert threshold_bytes == 64 * 1024 * 1024

    # Keep buf alive through the sleep so RSS stays high.
    del buf


@pytest.mark.no_memory_limit
def test_no_memory_limit_marker_disables_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int]] = []

    def recorder(nodeid: str, rss_bytes: int, threshold_bytes: int) -> None:
        calls.append((nodeid, rss_bytes, threshold_bytes))

    monkeypatch.setattr(tests_conftest, "_watchdog_kill_action", recorder)

    buf = bytearray(256 * 1024 * 1024)
    for i in range(0, len(buf), 4096):
        buf[i] = 1

    time.sleep(1.5)

    assert calls == [], (
        f"watchdog should be inactive for no_memory_limit tests, got {calls}"
    )

    del buf


def test_default_threshold_applied() -> None:
    from tests.conftest import _DEFAULT_WATCHDOG_MB

    assert _watchdog_state["threshold_bytes"] == _DEFAULT_WATCHDOG_MB * 1024 * 1024
