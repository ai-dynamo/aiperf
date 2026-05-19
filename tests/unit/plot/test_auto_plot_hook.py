# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the auto-plot post-run hook (warn vs strict semantics)."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.cli_runner import CompletedRun
from aiperf.plot.auto_plot import build_auto_plot_callback


class TestBuildAutoPlotCallback:
    """build_auto_plot_callback returns an OnComplete that wraps run_plot_controller."""

    def test_callback_invokes_run_plot_controller_with_artifact_path(
        self, tmp_path: Path
    ):
        callback = build_auto_plot_callback(plot_required=False)

        with patch("aiperf.plot.auto_plot.run_plot_controller") as mock_runner:
            callback(CompletedRun(artifact_dir=tmp_path))

        mock_runner.assert_called_once_with(paths=[str(tmp_path)], config=None)

    def test_warn_mode_swallows_exception_and_logs_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        callback = build_auto_plot_callback(plot_required=False)

        with (
            caplog.at_level(logging.WARNING, logger="aiperf.plot.auto_plot"),
            patch(
                "aiperf.plot.auto_plot.run_plot_controller",
                side_effect=RuntimeError("kaleido missing"),
            ),
        ):
            # Must not raise.
            callback(CompletedRun(artifact_dir=tmp_path))

        # The warning mentions the artifact dir so the user can re-run.
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, "expected at least one WARNING record"
        rendered = warning_records[0].getMessage()
        assert str(tmp_path) in rendered
        assert "aiperf plot" in rendered

    def test_strict_mode_reraises_underlying_exception(self, tmp_path: Path):
        callback = build_auto_plot_callback(plot_required=True)

        with (
            patch(
                "aiperf.plot.auto_plot.run_plot_controller",
                side_effect=RuntimeError("kaleido missing"),
            ),
            pytest.raises(RuntimeError, match="kaleido missing"),
        ):
            callback(CompletedRun(artifact_dir=tmp_path))

    def test_strict_mode_does_not_swallow_typeerror(self, tmp_path: Path):
        # Catches `Exception`, so any non-system exception should re-raise
        # under plot_required=True. Use TypeError to exercise that.
        callback = build_auto_plot_callback(plot_required=True)

        with (
            patch(
                "aiperf.plot.auto_plot.run_plot_controller",
                side_effect=TypeError("bad signature"),
            ),
            pytest.raises(TypeError, match="bad signature"),
        ):
            callback(CompletedRun(artifact_dir=tmp_path))

    def test_warn_mode_no_warning_when_runner_succeeds(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        callback = build_auto_plot_callback(plot_required=False)

        with (
            caplog.at_level(logging.WARNING, logger="aiperf.plot.auto_plot"),
            patch("aiperf.plot.auto_plot.run_plot_controller"),
        ):
            callback(CompletedRun(artifact_dir=tmp_path))

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records == []
