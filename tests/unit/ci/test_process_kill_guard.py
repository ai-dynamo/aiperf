# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _ProcessGroupKillGuard and the timeout killer factory."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.ci.test_docs_end_to_end.test_runner import (
    _make_process_group_timeout_killer,
    _ProcessGroupKillGuard,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="os.killpg not available on Windows"
)


def test_kill_guard_mark_finished_prevents_kill():
    killed = []
    guard = _ProcessGroupKillGuard()
    guard.mark_finished()
    proc = SimpleNamespace(pid=12345, poll=lambda: None)

    with patch.object(os, "killpg", lambda pid, sig: killed.append((pid, sig))):
        killer = _make_process_group_timeout_killer(
            proc=proc, test_num=1, server_name="test-server", guard=guard
        )
        killer()

    assert killed == []


def test_kill_guard_kills_running_process():
    killed = []
    guard = _ProcessGroupKillGuard()
    proc = SimpleNamespace(pid=12345, poll=lambda: None)

    with patch.object(os, "killpg", lambda pid, sig: killed.append((pid, sig))):
        killer = _make_process_group_timeout_killer(
            proc=proc, test_num=1, server_name="test-server", guard=guard
        )
        killer()

    assert len(killed) == 1
    assert killed[0][0] == 12345


def test_kill_guard_idempotent_double_call():
    killed = []
    guard = _ProcessGroupKillGuard()
    proc = SimpleNamespace(pid=12345, poll=lambda: None)

    with patch.object(os, "killpg", lambda pid, sig: killed.append((pid, sig))):
        killer = _make_process_group_timeout_killer(
            proc=proc, test_num=1, server_name="test-server", guard=guard
        )
        killer()
        killer()

    # Second call must be a no-op — guard tracks that kill already happened
    assert len(killed) == 1


def test_kill_guard_skips_already_exited_process():
    killed = []
    guard = _ProcessGroupKillGuard()
    proc = SimpleNamespace(pid=12345, poll=lambda: 0)  # already exited

    with patch.object(os, "killpg", lambda pid, sig: killed.append((pid, sig))):
        killer = _make_process_group_timeout_killer(
            proc=proc, test_num=1, server_name="test-server", guard=guard
        )
        killer()

    assert killed == []
