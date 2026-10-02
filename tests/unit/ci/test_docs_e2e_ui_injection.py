# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the docs-e2e runner's ``--ui-type`` injection.

The runner forces a non-interactive UI on every tagged command. Doing that
unconditionally collides with any guide that teaches ``--ui-type`` itself:
cyclopts rejects the repeated parameter and exits 1, which silently caps how
much of ``docs/`` can ever be covered by the end-to-end test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pytest import param

HARNESS = Path(__file__).resolve().parents[3] / "tests/ci/test_docs_end_to_end"
sys.path.insert(0, str(HARNESS))

from test_runner import inject_ui_type  # noqa: E402


@pytest.mark.parametrize(
    "command",
    [
        param("aiperf profile --model m --url localhost:8000", id="plain"),
        param("aiperf profile --model m --ui-types x", id="ui-types-is-a-different-flag"),
        param("aiperf profile --model my--ui-type", id="embedded-in-a-value"),
        param("aiperf profile --no-ui-type-thing x", id="longer-token"),
    ],
)  # fmt: skip
def test_inject_ui_type_adds_the_flag_when_absent(command: str) -> None:
    assert inject_ui_type(command, "simple").startswith(
        "aiperf profile --ui-type simple"
    )


@pytest.mark.parametrize(
    "command",
    [
        param("aiperf profile --model m --ui-type none", id="ui-type-space"),
        param("aiperf profile --model m --ui-type=none", id="ui-type-equals"),
        param("aiperf profile --model m --ui simple", id="ui-alias"),
        param("aiperf profile --model m \\\n  --ui-type dashboard", id="line-continuation"),
    ],
)  # fmt: skip
def test_inject_ui_type_defers_to_an_explicit_choice(command: str) -> None:
    assert inject_ui_type(command, "simple") == command
    assert command.count("--ui") == 1


def test_injected_command_never_repeats_the_parameter() -> None:
    """cyclopts exits 1 on a repeated parameter, so this is the load-bearing claim."""
    for command in (
        "aiperf profile --model m",
        "aiperf profile --model m --ui-type none",
        "aiperf profile --model m --ui simple",
    ):
        assert inject_ui_type(command, "simple").count("--ui-type") <= 1
