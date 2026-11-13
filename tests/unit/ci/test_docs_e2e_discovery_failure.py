# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed probe must not be mistaken for a successful empty answer.

Both of these run inside the per-server loop, where a wrong answer is worse
than no answer: the container snapshot decides which containers get a log
follower attached under this server's name, and the fixture write decides
whether this server runs at all. Reporting "nothing found" or raising, rather
than "could not tell", let one server's infrastructure failure corrupt or
abort every later server's run.
"""

from __future__ import annotations

import subprocess

import pytest
from pytest import param


@pytest.mark.parametrize(
    "outcome",
    [
        param(subprocess.TimeoutExpired(cmd="docker", timeout=30), id="timeout"),
        param(OSError("docker not on PATH"), id="docker-missing"),
    ],
)  # fmt: skip
def test_a_failed_snapshot_is_unavailable_not_empty(monkeypatch, outcome) -> None:
    """An empty set would read as a valid baseline of zero containers."""
    from test_runner import EndToEndTestRunner

    runner = EndToEndTestRunner()

    def _boom(*args, **kwargs):
        raise outcome

    monkeypatch.setattr(subprocess, "run", _boom)
    assert runner._known_container_ids() is None


def test_a_nonzero_docker_ps_is_unavailable_not_empty(monkeypatch) -> None:
    """``check=False`` meant a failing docker ps returned empty stdout, which
    is indistinguishable from a host with no containers."""
    from test_runner import EndToEndTestRunner

    runner = EndToEndTestRunner()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="boom"),
    )
    assert runner._known_container_ids() is None


@pytest.mark.parametrize(
    "before,after",
    [
        param(None, {"pre-existing", "new"}, id="pre-snapshot-failed"),
        param({"pre-existing"}, None, id="post-snapshot-failed"),
    ],
)  # fmt: skip
def test_an_unavailable_snapshot_follows_nothing(monkeypatch, before, after) -> None:
    """With a failed pre-snapshot read as empty, every container already on the
    host looked newly started and picked up a follower under this server's
    name."""
    from test_runner import EndToEndTestRunner

    runner = EndToEndTestRunner()
    followed: list[str] = []
    monkeypatch.setattr(runner, "_known_container_ids", lambda: after)
    monkeypatch.setattr(
        runner,
        "_stream_container_logs",
        lambda cid, name: followed.append(cid),
    )

    threads = runner._follow_detached_containers(before, "demo")

    assert threads == []
    assert followed == []


def test_a_successful_snapshot_still_follows_new_containers(monkeypatch) -> None:
    """The sentinel must not suppress the case this capture exists for."""
    from test_runner import EndToEndTestRunner

    runner = EndToEndTestRunner()
    followed: list[str] = []
    monkeypatch.setattr(runner, "_known_container_ids", lambda: {"pre-existing", "new"})
    monkeypatch.setattr(
        runner,
        "_stream_container_logs",
        lambda cid, name: followed.append(cid),
    )

    runner._follow_detached_containers({"pre-existing"}, "demo")
    for thread in runner._follow_detached_containers({"pre-existing"}, "demo"):
        thread.join(timeout=5)

    assert followed == ["new", "new"]


@pytest.mark.parametrize(
    "outcome",
    [
        param(subprocess.TimeoutExpired(cmd="docker", timeout=30), id="timeout"),
        param(OSError("exec failed"), id="oserror"),
    ],
)  # fmt: skip
def test_a_fixture_write_failure_returns_false(monkeypatch, outcome) -> None:
    """Raising here escapes the per-server loop, so one guide's unwritable
    fixture takes every later server's commands down with it."""
    from data_types import FileFixture, Server
    from test_runner import EndToEndTestRunner

    runner = EndToEndTestRunner()
    runner.aiperf_container_id = "abc123"
    server = Server(
        name="demo",
        setup_command=None,
        health_check_command=None,
        aiperf_commands=[],
        files=[FileFixture("cfg.yaml", "a: 1", "docs/demo.md", 10)],
    )

    def _boom(*args, **kwargs):
        raise outcome

    monkeypatch.setattr(subprocess, "run", _boom)

    assert runner._materialize_files(server) is False
