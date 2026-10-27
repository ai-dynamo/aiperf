# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A server's containers must be stopped however its test ends.

The teardown used to sit at the bottom of ``_test_server``, so every early
exit skipped it and left the engine holding its port. Each documented group
binds 8000, so in an ``--all-servers`` run the next server then failed with
``Bind for 0.0.0.0:8000 failed: port is already allocated`` -- a docker
networking error naming neither the real failure nor the server that caused
it. CI's single-server shards hide the interaction entirely.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest
from pytest import param

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/ci/test_docs_end_to_end"))

import test_runner as TR  # noqa: E402
from data_types import Command, Server  # noqa: E402


def _command(text: str) -> Command:
    return Command(tag_name="t", command=text, file_path="f", start_line=1, end_line=2)


class _FakeProcess:
    """Enough of ``Popen`` for the setup and health-check call sites."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.stdout = self

    def readline(self) -> str:
        return ""

    def poll(self) -> int:
        return self.returncode

    def wait(self, *args, **kwargs) -> int:
        return self.returncode


def _issued_commands(monkeypatch, *, health_rc: int, fixtures_ok: bool) -> list[str]:
    """Run one server end to end against fakes, returning the shell commands."""
    issued: list[str] = []
    runner = TR.EndToEndTestRunner()
    runner.aiperf_container_id = "aiperf-test-probe"
    runner.stop_log_monitoring = threading.Event()
    runner.log_monitoring_thread = None
    runner.setup_process = None
    monkeypatch.setattr(runner, "_known_container_ids", lambda: set())
    monkeypatch.setattr(runner, "_monitor_server_logs", lambda process, name: None)
    monkeypatch.setattr(runner, "_materialize_files", lambda server: fixtures_ok)

    def fake_popen(cmd, *args, **kwargs):
        text = cmd if isinstance(cmd, str) else " ".join(cmd)
        issued.append(text)
        return _FakeProcess(0 if "setup" in text else health_rc)

    def fake_run(cmd, *args, **kwargs):
        issued.append(cmd if isinstance(cmd, str) else " ".join(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(TR.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(TR.subprocess, "run", fake_run)

    server = Server(
        name="probe",
        setup_command=_command("setup cmd"),
        health_check_command=_command("health cmd"),
        aiperf_commands=[_command("aiperf profile --model m")],
    )
    runner._test_server(server)
    return issued


@pytest.mark.parametrize(
    "health_rc,fixtures_ok",
    [
        param(1, True, id="health-check-fails"),
        param(0, False, id="fixture-write-fails"),
        param(0, True, id="everything-succeeds"),
    ],
)  # fmt: skip
def test_server_containers_are_stopped_on_every_exit_path(
    monkeypatch, health_rc, fixtures_ok
) -> None:
    issued = _issued_commands(monkeypatch, health_rc=health_rc, fixtures_ok=fixtures_ok)

    assert any("docker stop" in command for command in issued)


def test_teardown_never_stops_the_aiperf_container(monkeypatch) -> None:
    """It outlives every server; stopping it would break the next one."""
    issued = _issued_commands(monkeypatch, health_rc=1, fixtures_ok=True)

    stop = next(command for command in issued if "docker stop" in command)
    assert "grep -v '^aiperf-test-probe$'" in stop
