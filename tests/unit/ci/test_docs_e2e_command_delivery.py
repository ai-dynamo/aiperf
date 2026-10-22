# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A command must not be able to eat the rest of its own block.

The runner used to deliver the tagged block to ``bash -s`` over stdin. That
left the script on the same stream the commands read from, so the first
command that reads stdin consumed everything after it: ``cat`` followed by
another command printed that command instead of running it, and the block
still exited 0. A guide could then benchmark nothing at all while CI reported
a pass -- the one outcome this harness exists to prevent.

Heredocs are safe either way, because bash resolves them while parsing, which
is why no tagged guide ever tripped this. A runtime reader does trip it, and
``aiperf chat`` reads stdin today.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aiperf.common.constants import IS_WINDOWS

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/ci/test_docs_end_to_end"))

from test_runner import build_command_exec_argv  # noqa: E402

# A block whose first command reads stdin, followed by one that must still run.
SWALLOWING_BLOCK = "cat\necho AFTER_RAN"

needs_bash = pytest.mark.skipif(
    IS_WINDOWS, reason="docs-e2e commands run in a Linux container; needs bash"
)


def test_the_command_is_an_argv_element_not_a_shell_string() -> None:
    """Interpolating it into a shell string is what mangles quoted JSON."""
    argv = build_command_exec_argv("container", """--extra-inputs '{"a": 0}'""")

    assert argv[-2:] == ["-c", """--extra-inputs '{"a": 0}'"""]
    assert "-s" not in argv, "a script on stdin is readable by the commands it runs"


def test_stdin_is_not_attached_to_the_exec() -> None:
    """``-i`` is what would hand the commands an open stdin to consume."""
    assert "-i" not in build_command_exec_argv("container", "aiperf profile")


@needs_bash
def test_a_command_that_reads_stdin_does_not_swallow_the_block() -> None:
    """The real shell, invoked the way the runner invokes it."""
    argv = build_command_exec_argv("unused", SWALLOWING_BLOCK)
    # Drop the docker prefix; the shell behaviour is what is under test.
    proc = subprocess.run(
        argv[argv.index("bash") :],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "AFTER_RAN", (
        "the command after the stdin reader must execute, not be printed"
    )


@needs_bash
def test_stdin_delivery_is_what_broke() -> None:
    """Pins the old behaviour so the reason for argv delivery stays visible."""
    proc = subprocess.run(
        ["bash", "-s"], input=SWALLOWING_BLOCK, capture_output=True, text=True
    )

    assert proc.returncode == 0, "the silent part: a swallowed block still exits 0"
    assert proc.stdout.strip() == "echo AFTER_RAN", (
        "stdin delivery lets `cat` read the rest of the script as data"
    )


@needs_bash
def test_a_heredoc_still_works() -> None:
    """The pattern most tagged guides actually use; resolved while parsing."""
    argv = build_command_exec_argv(
        "unused", "cat << 'EOF'\nline one\nline two\nEOF\necho DONE"
    )
    proc = subprocess.run(
        argv[argv.index("bash") :],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["line", "one", "line", "two", "DONE"]
