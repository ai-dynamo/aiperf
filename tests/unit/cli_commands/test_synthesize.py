# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the `aiperf synthesize` CLI command error boundary and constraints."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.cli_commands.synthesize import app, synthesize


class TestSynthesizeErrorBoundary:
    """The command surfaces clean errors + exit 1 for bad config inputs."""

    def test_malformed_config_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A malformed --config JSON renders a clean panel and exits 1."""
        bad = tmp_path / "bad.json"
        bad.write_text("{ this is not valid json")

        with pytest.raises(SystemExit) as exc:
            synthesize(
                "agentic-code",
                config=str(bad),
                num_sessions=2,
                output=tmp_path,
            )

        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "Error Synthesizing Dataset" in err

    def test_missing_config_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non-existent --config path renders a clean panel and exits 1."""
        missing = tmp_path / "does_not_exist.json"

        with pytest.raises(SystemExit) as exc:
            synthesize(
                "agentic-code",
                config=str(missing),
                num_sessions=2,
                output=tmp_path,
            )

        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower()


class TestSynthesizeNumericConstraints:
    """Out-of-range numeric flags are rejected at the CLI boundary."""

    @pytest.mark.parametrize(
        "flag",
        ["--num-sessions", "--max-isl", "--max-osl"],
    )  # fmt: skip
    def test_zero_is_rejected(
        self, flag: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A zero value for a positive-only flag exits 1 with a clean message."""
        with pytest.raises(SystemExit) as exc:
            app(["agentic-code", flag, "0"])

        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert flag in err
        assert ">= 1" in err

    def test_negative_num_sessions_is_rejected(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A negative --num-sessions exits 1 with a clean message."""
        with pytest.raises(SystemExit) as exc:
            app(["agentic-code", "--num-sessions", "-5"])

        assert exc.value.code == 1
        assert ">= 1" in capsys.readouterr().err
