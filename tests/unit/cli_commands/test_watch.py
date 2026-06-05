# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the watch CLI command."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from aiperf.cli_commands.watch import watch


class TestWatchCommand:
    """Tests for watch() CLI forwarding."""

    def test_forwards_config_mode_arguments(self, monkeypatch) -> None:
        runner = MagicMock()
        monkeypatch.setitem(
            sys.modules,
            "aiperf._tachometer",
            SimpleNamespace(run_tachometer_cli=runner),
        )

        watch(config=Path("/tmp/watch.toml"), local_dir=Path("/tmp/local"))

        runner.assert_called_once_with(
            [
                "--config",
                "/tmp/watch.toml",
                "--freq",
                "0.2",
                "--rows-per-parquet",
                "1000000",
                "--save-interval",
                "5",
                "--local-dir",
                "/tmp/local",
                "--sync-interval",
                "0",
            ]
        )

    def test_forwards_endpoint_mode_arguments(self, monkeypatch) -> None:
        runner = MagicMock()
        monkeypatch.setitem(
            sys.modules,
            "aiperf._tachometer",
            SimpleNamespace(run_tachometer_cli=runner),
        )

        watch(
            endpoints=["gpu=http://localhost:9400/metrics"],
            frequency=1.0,
            storage="/tmp/run",
            rows_per_parquet=10,
            save_interval_secs=2,
            filters=["dcgm"],
            sync_interval_secs=30,
        )

        runner.assert_called_once_with(
            [
                "--endpoint",
                "gpu=http://localhost:9400/metrics",
                "--freq",
                "1.0",
                "--storage",
                "/tmp/run",
                "--rows-per-parquet",
                "10",
                "--save-interval",
                "2",
                "--filter",
                "dcgm",
                "--sync-interval",
                "30",
            ]
        )


class TestWatchAppRegistration:
    """Tests for the Cyclopts App object registration."""

    def test_app_name(self) -> None:
        from aiperf.cli_commands.watch import app

        assert "watch" in app.name

    def test_root_app_registers_watch(self) -> None:
        from aiperf.cli import app

        assert "watch" in app._commands
