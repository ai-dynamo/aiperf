# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for dashboard-without-TTY fallback in build_logging_runtime.

When ``--ui dashboard`` is requested but stdout is not a TTY (subprocess
PIPE, shell redirection, CI capture), the Textual TUI issues console-setup
syscalls that block forever on Windows. The converter downgrades to
``--ui simple`` instead of hanging.
"""

from __future__ import annotations

import pytest

from aiperf.config.flags._converter_runtime import build_logging_runtime
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import UIType


class TestDashboardTtyFallback:
    """``--ui dashboard`` downgrades to SIMPLE when stdout is not a TTY."""

    def test_dashboard_without_tty_downgrades_to_simple(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("aiperf.common.utils.is_tty", lambda: False, raising=True)
        cli = CLIConfig(
            url="http://localhost:8000/test",
            model_names=["test-model"],
            ui_type=UIType.DASHBOARD,
        )

        _, runtime = build_logging_runtime(cli)

        assert runtime["ui"] == UIType.SIMPLE

    def test_dashboard_with_tty_stays_dashboard(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("aiperf.common.utils.is_tty", lambda: True, raising=True)
        cli = CLIConfig(
            url="http://localhost:8000/test",
            model_names=["test-model"],
            ui_type=UIType.DASHBOARD,
        )

        _, runtime = build_logging_runtime(cli)

        assert runtime["ui"] == UIType.DASHBOARD

    def test_simple_without_tty_stays_simple(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("aiperf.common.utils.is_tty", lambda: False, raising=True)
        cli = CLIConfig(
            url="http://localhost:8000/test",
            model_names=["test-model"],
            ui_type=UIType.SIMPLE,
        )

        _, runtime = build_logging_runtime(cli)

        assert runtime["ui"] == UIType.SIMPLE

    def test_unset_ui_without_tty_defaults_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Existing pre-branch behavior: unset UI + non-TTY -> NONE, not SIMPLE."""
        monkeypatch.setattr("aiperf.common.utils.is_tty", lambda: False, raising=True)
        cli = CLIConfig(
            url="http://localhost:8000/test",
            model_names=["test-model"],
        )

        _, runtime = build_logging_runtime(cli)

        assert runtime["ui"] == UIType.NONE
