# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sweep table is rendered to a string, so Rich cannot see the real console.

Rich substitutes box-drawing characters itself when it writes to a legacy
Windows console, but this table goes to a StringIO and then to the logger, so
that safety net never runs. The box style has to be chosen from the encoding of
the stream the log handler writes to.
"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aiperf.cli_runner._sweep_table import SweepTableLogger

_BOX_DRAWING = "\u2500\u2501"


@pytest.mark.parametrize(
    ("encoding", "expect_box_drawing"),
    [
        ("utf-8", True),
        ("UTF-8", True),
        (None, False),
        ("cp1252", False),
        ("not-a-codec", False),
    ],
)
def test_box_style_follows_the_stdout_encoding(
    monkeypatch: pytest.MonkeyPatch, encoding: str | None, expect_box_drawing: bool
) -> None:
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(encoding=encoding))

    style = str(SweepTableLogger._box_style())

    if expect_box_drawing:
        assert any(ch in style for ch in _BOX_DRAWING)
    else:
        assert style.isascii(), "a narrow sink needs a style it can encode"


@pytest.mark.parametrize("encoding", ["cp1252", None])
def test_a_non_ascii_parameter_name_does_not_cost_the_table(
    monkeypatch: pytest.MonkeyPatch, encoding: str | None
) -> None:
    """A ``variables.<name>`` sweep can carry any name; on a narrow sink it is
    substituted rather than failing the write and dropping the table."""
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(encoding=encoding))
    name = "t\u00e9mp\u2192k"
    plan = SimpleNamespace(
        variations=[SimpleNamespace(values={name: 1})],
        confidence_level=0.95,
        sweep=None,
    )
    logger = Mock()

    SweepTableLogger(plan, logger)(("key",), {"params": {name: 1}})

    logged = logger.info.call_args[0][0]
    logged.encode(encoding or "ascii")
    assert "mp" in logged
