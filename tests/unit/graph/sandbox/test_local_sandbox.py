# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local session sandbox: framing, isolation, timeout, lifecycle."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from pytest import param

from aiperf.graph.sandbox.local import LocalSessionSandbox


@pytest.mark.asyncio
async def test_runs_a_command_and_captures_stdout(tmp_path: Path) -> None:
    sandbox = LocalSessionSandbox(workspace=tmp_path)
    await sandbox.open()
    try:
        result = await sandbox.run("echo hello")
    finally:
        await sandbox.close()
    assert result.stdout.strip() == "hello"
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.duration_s > 0


@pytest.mark.asyncio
async def test_reports_nonzero_returncode(tmp_path: Path) -> None:
    sandbox = LocalSessionSandbox(workspace=tmp_path)
    await sandbox.open()
    try:
        result = await sandbox.run("exit 3")
    finally:
        await sandbox.close()
    assert result.returncode == 3


@pytest.mark.asyncio
async def test_commands_do_not_leak_state_between_calls(tmp_path: Path) -> None:
    """Recorded commands ran under per-call `bash -lc`, so a `cd` must not persist.

    A bare persistent shell would carry the directory change into the next
    command and silently diverge from the trajectory being replayed.
    """
    (tmp_path / "sub").mkdir()
    sandbox = LocalSessionSandbox(workspace=tmp_path)
    await sandbox.open()
    try:
        await sandbox.run("cd sub")
        result = await sandbox.run("pwd")
    finally:
        await sandbox.close()
    assert result.stdout.strip() == str(tmp_path)


@pytest.mark.asyncio
async def test_multiline_command_is_framed_correctly(tmp_path: Path) -> None:
    """Recorded PinchBench commands include heredocs spanning many lines."""
    sandbox = LocalSessionSandbox(workspace=tmp_path)
    await sandbox.open()
    try:
        result = await sandbox.run("cat > f.txt << 'EOF'\nline1\nline2\nEOF")
        readback = await sandbox.run("cat f.txt")
    finally:
        await sandbox.close()
    assert result.returncode == 0
    assert readback.stdout.strip().splitlines() == ["line1", "line2"]


@pytest.mark.asyncio
async def test_timeout_marks_result_and_keeps_session_usable(tmp_path: Path) -> None:
    sandbox = LocalSessionSandbox(workspace=tmp_path, default_timeout_s=0.5)
    await sandbox.open()
    try:
        timed = await sandbox.run("sleep 5")
        after = await sandbox.run("echo alive")
    finally:
        await sandbox.close()
    assert timed.timed_out is True
    assert after.stdout.strip() == "alive"
    # Recycling the session costs a kill grace plus a spawn; none of it belongs
    # in the reported duration, which is the metric this feature exists to emit.
    assert 0.5 <= timed.duration_s < 1.0


@pytest.mark.asyncio
async def test_stale_sentinel_cannot_desynchronise_the_session(tmp_path: Path) -> None:
    """A marker learned from one command must not terminate a later read.

    The sentinel is written into the session shell's stdin, so a command can
    discover it. If the token were fixed per session, echoing it would truncate
    the next command's output and leave the reader one command behind forever.
    """
    sandbox = LocalSessionSandbox(workspace=tmp_path)
    issued: list[str] = []
    mint = sandbox._next_sentinel

    def _record() -> str:
        token = mint()
        issued.append(token)
        return token

    sandbox._next_sentinel = _record  # type: ignore[method-assign]
    await sandbox.open()
    try:
        first = await sandbox.run("echo one")
        forged = issued[0]
        second = await sandbox.run(f"echo '{forged}0'; echo two")
        third = await sandbox.run("echo three")
    finally:
        await sandbox.close()
    assert first.stdout.strip() == "one"
    assert second.stdout.strip().splitlines() == [f"{forged}0", "two"]
    assert second.returncode == 0
    assert third.stdout.strip() == "three"


@pytest.mark.asyncio
async def test_run_before_open_raises(tmp_path: Path) -> None:
    sandbox = LocalSessionSandbox(workspace=tmp_path)
    with pytest.raises(RuntimeError, match="not open"):
        await sandbox.run("echo hi")


@pytest.mark.asyncio
async def test_close_is_idempotent(tmp_path: Path) -> None:
    sandbox = LocalSessionSandbox(workspace=tmp_path)
    await sandbox.open()
    await sandbox.close()
    await sandbox.close()


@pytest.mark.asyncio
async def test_command_without_trailing_newline_completes(tmp_path: Path) -> None:
    """A newline-less stdout must not swallow the sentinel and hang until timeout.

    The sentinel is matched at the start of a line, so before the framing
    emitted its own newline the sentinel landed on the same line as this
    command's output and never matched -- the command "took" the full timeout
    and recycled the session (destroying the container on the docker backend).
    """
    sandbox = LocalSessionSandbox(workspace=tmp_path, default_timeout_s=5.0)
    await sandbox.open()
    try:
        result = await sandbox.run("printf 'no-newline'")
    finally:
        await sandbox.close()
    assert result.timed_out is False
    assert result.returncode == 0
    assert result.stdout.startswith("no-newline")
    assert result.duration_s < 2.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,expected",
    [
        param("printf 'a\\n'", "a\n", id="single-line"),
        param("printf 'a\\nb\\n'", "a\nb\n", id="two-lines"),
        param("printf 'a\\n\\n\\n'", "a\n\n\n", id="trailing-blank-lines"),
        param("printf ''", "", id="empty"),
        param("printf '\\n'", "\n", id="only-a-newline"),
    ],
)  # fmt: skip
async def test_stdout_is_preserved_exactly(
    tmp_path: Path, command: str, expected: str
) -> None:
    """The framing newline is stripped exactly, so newline-terminated output round-trips byte for byte."""
    sandbox = LocalSessionSandbox(workspace=tmp_path)
    await sandbox.open()
    try:
        result = await sandbox.run(command)
    finally:
        await sandbox.close()
    assert result.stdout == expected


@pytest.mark.asyncio
async def test_timed_out_command_child_is_killed(tmp_path: Path) -> None:
    """The abandoned `bash -lc` child must die with the session, not keep burning CPU.

    A survivor competes with every LATER measured command, so the whole
    duration series after one timeout would be wrong. Checked by PID rather
    than by a sleep-then-look: the suite's auto-fixture makes `asyncio.sleep`
    instant, so a wait-for-the-marker test would pass without ever waiting.
    """
    pidfile = tmp_path / "child.pid"
    sandbox = LocalSessionSandbox(workspace=tmp_path, default_timeout_s=0.5)
    await sandbox.open()
    try:
        await sandbox.run(f"echo $$ > {pidfile}; sleep 300")
    finally:
        await sandbox.close()

    child_pid = int(pidfile.read_text().strip())
    # SIGKILL delivery is asynchronous; poll in REAL time (time.sleep, not
    # asyncio.sleep) for a bounded moment rather than racing it.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.05)
    pytest.fail(f"timed-out command's child {child_pid} survived the session recycle")


@pytest.mark.asyncio
async def test_dead_session_reports_the_cause_not_a_transport_error(
    tmp_path: Path,
) -> None:
    """A session argv the environment cannot hold open must name itself.

    The raw failure is `unable to perform operation on <WriteUnixTransport
    closed=True>`, which tells an operator nothing. The real cause is nearly
    always a task image with no `bash`, and the shell says so.
    """
    sandbox = LocalSessionSandbox(workspace=tmp_path)
    sandbox._session_argv = lambda: ["bash", "-c", "exit 42"]  # type: ignore[method-assign]
    await sandbox.open()
    try:
        with pytest.raises(RuntimeError, match="died"):
            await sandbox.run("echo hi")
    finally:
        await sandbox.close()
