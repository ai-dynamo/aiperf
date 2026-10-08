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


@pytest.mark.parametrize(
    "command",
    [
        param(
            """aiperf profile --model m --extra-inputs '{"note": "pass --ui-type none here"}'""",
            id="quoted-flag-is-data-not-an-option",
        ),
        param(
            """aiperf profile --model m --extra-inputs '{"cmd": "aiperf profile"}'""",
            id="quoted-command-is-data-not-an-invocation",
        ),
    ],
)  # fmt: skip
def test_quoted_text_does_not_change_ui_injection(command: str) -> None:
    """Guides pass JSON payloads; their contents are data, not shell options.

    Reading a quoted ``--ui-type`` as an explicit choice leaves the interactive
    UI on and the command hangs in CI. Rewriting a quoted ``aiperf profile``
    corrupts the payload the guide meant to send.
    """
    injected = inject_ui_type(command)

    assert injected.count("--ui-type") == command.count("--ui-type") + 1
    assert injected.startswith("aiperf profile --ui-type simple ")
    # Everything after the invocation is untouched.
    assert injected.endswith(command[len("aiperf profile") :])


@pytest.mark.parametrize(
    "command",
    [
        param(
            "# aiperf profile --model commented-out\naiperf profile --model m",
            id="commented-variant-above-the-real-command",
        ),
        param(
            "  # aiperf profile --model x\naiperf profile --model m",
            id="indented-comment",
        ),
    ],
)  # fmt: skip
def test_a_commented_command_is_not_the_invocation(command: str) -> None:
    """Guides show a commented variant above the command they actually run.

    Injecting into the comment leaves the real invocation interactive, and it
    then hangs until the watchdog kills it.
    """
    injected = inject_ui_type(command)

    real_line = injected.splitlines()[-1]
    assert real_line.startswith("aiperf profile --ui-type simple ")
    assert "--ui-type" not in injected.splitlines()[0]


def test_a_hash_inside_quotes_is_not_a_comment() -> None:
    injected = inject_ui_type('echo "# not a comment" && aiperf profile --model m')
    assert "aiperf profile --ui-type simple --model m" in injected
