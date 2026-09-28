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

import pytest

from aiperf.cli_runner._sweep_table import SweepTableLogger

_BOX_DRAWING = "\u2500\u2501"


@pytest.mark.parametrize(
    ("encoding", "expect_box_drawing"),
    [
        ("utf-8", True),
        ("UTF-8", True),
        (None, True),
        ("cp1252", False),
        ("ascii", False),
        ("latin-1", False),
        ("not-a-codec", False),
    ],
)
def test_box_style_follows_the_stdout_encoding(
    monkeypatch: pytest.MonkeyPatch, encoding: str | None, expect_box_drawing: bool
) -> None:
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(encoding=encoding))

    box = SweepTableLogger._box_style()

    uses_box_drawing = any(ch in str(box) for ch in _BOX_DRAWING)
    assert uses_box_drawing is expect_box_drawing


def test_the_chosen_style_is_encodable_by_that_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the choice: whatever comes back must survive the console."""
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(encoding="cp1252"))

    str(SweepTableLogger._box_style()).encode("cp1252")
