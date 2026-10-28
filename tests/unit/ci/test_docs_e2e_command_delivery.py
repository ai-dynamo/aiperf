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

import pytest

from aiperf.common.constants import IS_WINDOWS

# A block whose first command reads stdin, followed by one that must still run.
SWALLOWING_BLOCK = "cat\necho AFTER_RAN"

needs_bash = pytest.mark.skipif(
    IS_WINDOWS, reason="docs-e2e commands run in a Linux container; needs bash"
)


def test_the_command_is_an_argv_element_not_a_shell_string() -> None:
    """Interpolating it into a shell string is what mangles quoted JSON."""
    from constants import AIPERF_COMMAND_TIMEOUT
    from test_runner import build_command_exec_argv

    argv = build_command_exec_argv(
        "container", """--extra-inputs '{"a": 0}'""", AIPERF_COMMAND_TIMEOUT
    )

    assert argv[-2:] == ["-c", """--extra-inputs '{"a": 0}'"""]
    assert "-s" not in argv, "a script on stdin is readable by the commands it runs"


def test_stdin_is_not_attached_to_the_exec() -> None:
    """``-i`` is what would hand the commands an open stdin to consume."""
    from constants import AIPERF_COMMAND_TIMEOUT
    from test_runner import build_command_exec_argv

    assert "-i" not in build_command_exec_argv(
        "container", "aiperf profile", AIPERF_COMMAND_TIMEOUT
    )


@needs_bash
def test_a_command_that_reads_stdin_does_not_swallow_the_block() -> None:
    """The real shell, invoked the way the runner invokes it."""
    from constants import AIPERF_COMMAND_TIMEOUT
    from test_runner import build_command_exec_argv

    argv = build_command_exec_argv("unused", SWALLOWING_BLOCK, AIPERF_COMMAND_TIMEOUT)
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
    from constants import AIPERF_COMMAND_TIMEOUT
    from test_runner import build_command_exec_argv

    argv = build_command_exec_argv(
        "unused",
        "cat << 'EOF'\nline one\nline two\nEOF\necho DONE",
        AIPERF_COMMAND_TIMEOUT,
    )
    proc = subprocess.run(
        argv[argv.index("bash") :],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["line", "one", "line", "two", "DONE"]


def test_the_deadline_is_enforced_inside_the_container() -> None:
    """Killing the host-side `docker exec` client leaves its in-container child
    running: verified with real containers, the exec'd process survived
    `killpg(SIGKILL)` and kept the port and artifact paths busy for the next
    command. The deadline has to travel into the container with the command."""
    from test_runner import build_command_exec_argv

    argv = build_command_exec_argv("container", "aiperf profile", 900)
    assert "timeout" in argv, "no in-container deadline"
    # Indexed from the end: a container literally named "timeout" would
    # otherwise make argv.index() find the wrong element.
    deadline = int(argv[-4])
    # Strictly later than the host watchdog's, or the two race and a timeout
    # reports as a bare exit 124 instead of "exceeded 900s timeout".
    assert deadline > 900, "the in-container deadline must not race the watchdog"
    assert any(a.startswith("--kill-after=") for a in argv), (
        "a command that ignores SIGTERM would outlive its own deadline"
    )
    # The command still arrives as one argv element after `bash -c`.
    assert argv[-1] == "aiperf profile"
    assert argv[-2] == "-c"


def test_a_per_command_timeout_reaches_the_container() -> None:
    """`timeout=` on a run tag must move the in-container deadline too, or the
    guide's override applies only to the host watchdog."""
    from data_types import Command
    from test_runner import build_command_exec_argv, resolve_command_timeout

    cmd = Command(
        tag_name="aiperf-run-demo-endpoint-server",
        command="aiperf profile",
        file_path="g.md",
        start_line=1,
        end_line=3,
        timeout=42,
    )
    argv = build_command_exec_argv(
        "container", cmd.command, resolve_command_timeout(cmd)
    )
    assert int(argv[-4]) > 42, (
        "a per-command timeout= must move the in-container deadline with it"
    )


def test_the_guard_reports_which_side_won_the_race() -> None:
    """A watchdog kill leaves the command alive inside the container until its
    own later deadline. The reader has to know that happened, or it starts the
    next documented command against a server the previous one is still
    loading, and blames the overlap on the wrong command."""
    from test_runner import _ProcessGroupKillGuard

    class _Running:
        pid = 1

        def poll(self):  # noqa: ANN201 - test double
            return None

    # Watchdog fired first: the reader must be told.
    guard = _ProcessGroupKillGuard()
    assert guard.mark_killing_if_running(_Running()) is True
    assert guard.mark_finished() is True, "a watchdog kill was not reported"

    # Command exited on its own: nothing is left running, so no wait.
    guard = _ProcessGroupKillGuard()
    assert guard.mark_finished() is False
    assert guard.mark_killing_if_running(_Running()) is False, (
        "the watchdog must not kill a process the reader already reaped"
    )
