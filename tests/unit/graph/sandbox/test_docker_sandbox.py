# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docker session sandbox argv and lifecycle, without requiring a Docker daemon."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aiperf.common.environment import Environment
from aiperf.graph.sandbox import docker as docker_module
from aiperf.graph.sandbox.docker import DockerSessionSandbox


def test_start_argv_mounts_workspace_and_pins_network(tmp_path: Path) -> None:
    sandbox = DockerSessionSandbox(
        image="agent-trace-pinchbench:latest",
        workspace=tmp_path,
        network="none",
        container_name="aiperf-trace-1",
    )
    argv = sandbox.start_argv()
    assert argv[:3] == ["docker", "run", "-d"]
    assert "--rm" in argv
    assert "--name" in argv and "aiperf-trace-1" in argv
    assert "--network" in argv and "none" in argv
    assert f"{tmp_path}:/workspace" in argv
    assert "agent-trace-pinchbench:latest" in argv


@pytest.mark.asyncio
async def test_run_spawns_fresh_exec_per_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each run() must spawn a fresh docker exec — no shared bash session."""
    seen_argv: list[list[str]] = []

    class _DoneProc:
        pid = -1
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"hello\n", b""

    async def _fake_exec(*argv: str, **_kwargs: object) -> _DoneProc:
        seen_argv.append(list(argv))
        return _DoneProc()

    monkeypatch.setattr(docker_module.asyncio, "create_subprocess_exec", _fake_exec)
    sandbox = DockerSessionSandbox(
        image="img", workspace=tmp_path, container_name="aiperf-trace-2"
    )

    result = await sandbox.run("echo hi")

    assert result.stdout == "hello\n"
    assert len(seen_argv) == 1
    argv = seen_argv[0]
    assert argv[:2] == ["docker", "exec"]
    assert "-w" in argv and "/workspace" in argv
    assert "aiperf-trace-2" in argv
    assert "echo hi" in argv
    # No persistent session arg (-i) — each exec is self-contained.
    assert "-i" not in argv


@pytest.mark.asyncio
async def test_run_uses_custom_cwd_and_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_argv: list[list[str]] = []

    class _DoneProc:
        pid = -1
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec(*argv: str, **_kwargs: object) -> _DoneProc:
        seen_argv.append(list(argv))
        return _DoneProc()

    monkeypatch.setattr(docker_module.asyncio, "create_subprocess_exec", _fake_exec)
    sandbox = DockerSessionSandbox(
        image="img",
        workspace=tmp_path,
        container_name="aiperf-trace-settings",
        cwd="/testbed",
        interpreter=("bash", "-c"),
    )

    await sandbox.run("echo hi")

    assert seen_argv == [
        [
            "docker",
            "exec",
            "-w",
            "/testbed",
            "aiperf-trace-settings",
            "bash",
            "-c",
            "echo hi",
        ]
    ]


@pytest.mark.asyncio
async def test_fresh_exec_timeout_retires_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    class _TimedOutExec:
        pid = -1
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            raise TimeoutError

        async def wait(self) -> int:
            self.returncode = -9
            return -9

    class _RemovedContainer:
        pid = -1
        returncode = 0

        async def wait(self) -> int:
            return 0

    async def _fake_exec(*argv: str, **_kwargs: object) -> object:
        calls.append(list(argv))
        if argv[:2] == ("docker", "exec"):
            return _TimedOutExec()
        return _RemovedContainer()

    monkeypatch.setattr(docker_module.asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(
        DockerSessionSandbox,
        "_kill_process_group",
        staticmethod(lambda proc: None),
    )
    sandbox = DockerSessionSandbox(
        image="img", workspace=tmp_path, container_name="aiperf-timeout"
    )

    result = await sandbox.run("sleep forever")

    assert result.timed_out is True
    assert ["docker", "rm", "-f", "aiperf-timeout"] in calls
    with pytest.raises(RuntimeError, match="retired"):
        await sandbox.run("echo unsafe")


def test_persistent_session_argv_execs_into_named_container(tmp_path: Path) -> None:
    """In persistent-session mode the inherited session argv must target the container."""
    sandbox = DockerSessionSandbox(
        image="img",
        workspace=tmp_path,
        container_name="aiperf-trace-ps",
        persistent_session=True,
    )
    argv = sandbox._session_argv()
    assert argv[:3] == ["docker", "exec", "-i"]
    assert "aiperf-trace-ps" in argv
    assert "bash" in argv


def test_container_name_is_generated_when_absent(tmp_path: Path) -> None:
    a = DockerSessionSandbox(image="img", workspace=tmp_path)
    b = DockerSessionSandbox(image="img", workspace=tmp_path)
    assert a.container_name != b.container_name
    assert a.container_name.startswith("aiperf-tool-")


def test_stop_argv_removes_the_container(tmp_path: Path) -> None:
    sandbox = DockerSessionSandbox(
        image="img", workspace=tmp_path, container_name="aiperf-trace-3"
    )
    assert sandbox.stop_argv() == ["docker", "rm", "-f", "aiperf-trace-3"]


def test_commands_run_against_the_container_mount_point(tmp_path: Path) -> None:
    sandbox = DockerSessionSandbox(image="img", workspace=tmp_path)
    assert sandbox._workspace_in_sandbox == "/workspace"


@pytest.mark.asyncio
async def test_close_gives_up_when_container_removal_stalls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A wedged daemon must not stall the benchmark loop; the removal subprocess is killed, reaped, and reported."""
    monkeypatch.setattr(Environment.GRAPH_TOOL, "CONTAINER_STOP_TIMEOUT", 0.05)
    seen_argv: list[list[str]] = []

    class _StallingProc:
        """Stalls its FIRST wait (the bounded one) and resolves the reaping one."""

        pid = -1
        returncode: int | None = None

        def __init__(self) -> None:
            self._waits = 0

        async def wait(self) -> int:
            self._waits += 1
            if self._waits == 1:
                # Never touches asyncio.sleep -- the suite's auto-fixture makes
                # sleeps instant, which would mask an unbounded await here.
                await asyncio.Event().wait()
            self.returncode = -9
            return -9

    async def _fake_exec(*argv: str, **_kwargs: object) -> _StallingProc:
        seen_argv.append(list(argv))
        return _StallingProc()

    killed: list[object] = []
    monkeypatch.setattr(docker_module.asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(
        DockerSessionSandbox,
        "_kill_process_group",
        staticmethod(lambda proc: killed.append(proc)),
    )
    sandbox = DockerSessionSandbox(
        image="img", workspace=tmp_path, container_name="aiperf-trace-4"
    )

    # The outer bound fails the test rather than hanging the suite if close()
    # is unbounded.
    with caplog.at_level("WARNING"):
        await asyncio.wait_for(sandbox.close(), timeout=2.0)

    assert seen_argv == [["docker", "rm", "-f", "aiperf-trace-4"]]
    assert len(killed) == 1
    assert killed[0].returncode == -9, "the killed removal subprocess must be reaped"
    assert "aiperf-trace-4" in caplog.text


@pytest.mark.asyncio
async def test_removal_subprocess_gets_its_own_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`docker rm -f` must lead its own group, or killing it would signal the benchmark's own group."""
    seen_kwargs: list[dict] = []

    class _DoneProc:
        pid = -1
        returncode = 0

        async def wait(self) -> int:
            return 0

    async def _fake_exec(*_argv: str, **kwargs: object) -> _DoneProc:
        seen_kwargs.append(dict(kwargs))
        return _DoneProc()

    monkeypatch.setattr(docker_module.asyncio, "create_subprocess_exec", _fake_exec)
    sandbox = DockerSessionSandbox(image="img", workspace=tmp_path)

    await sandbox.close()

    assert seen_kwargs[0]["start_new_session"] is True
