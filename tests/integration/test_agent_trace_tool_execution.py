# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real tool execution end to end: lowering, dispatch, and both sandboxes composed.

Every other test in this area stands one layer up on a mock. These run actual
processes, so they are the only evidence that the recorded command shapes
survive the transport -- notably the heredoc form, which a line-oriented
transport mangles, and container teardown, which a leaked container outlives.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from aiperf.dataset.graph.adapters.mini_swe_agent_trace.recording import (
    from_mini_swe_agent_trace,
)
from aiperf.dataset.graph.models import ToolNode
from aiperf.graph.placement import PlacementContext
from aiperf.graph.sandbox.docker import DockerSessionSandbox
from aiperf.graph.sandbox.local import LocalSessionSandbox
from aiperf.graph.tool_dispatch.protocols import ToolDispatchRequest
from aiperf.graph.tool_dispatch.sandbox_dispatcher import SandboxToolDispatcher

IMAGE = "agent-trace-pinchbench:latest"
BASE_TS = 1_700_000_000.0

# The exact shapes the PinchBench `task_files` trace records: a directory
# creation, a quoted heredoc carrying a multi-line file, and a read-back.
MKDIR_COMMAND = "mkdir -p src"
HEREDOC_COMMAND = (
    "cat > src/main.py << 'EOF'\n"
    "def main():\n"
    '    print("Hello, World!")\n'
    "\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    main()\n"
    "EOF"
)
HEREDOC_FILE_TEXT = (
    "def main():\n"
    '    print("Hello, World!")\n'
    "\n"
    "\n"
    'if __name__ == "__main__":\n'
    "    main()\n"
)


def _docker_available() -> bool:
    """Collection-time probe; synchronous by necessity, never inside a test."""
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return probe.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_available(), reason=f"docker and {IMAGE} required"
)


def _model_call(event_id: int, step: int, start: float, dur: float) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "model_call",
        "timestamp": start + dur,
        "step": step,
        "duration_ns": int(dur * 1e9),
        "provider_request": {
            "messages": [{"role": "user", "content": "write main.py"}],
            "model": "openai/m",
        },
        "response_message": {
            "role": "assistant",
            "extra": {
                "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
            },
        },
    }


def _tool_call(
    event_id: int, step: int, start: float, dur: float, command: str
) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": "tool_call",
        "timestamp": start + dur,
        "step": step,
        "action_index": 0,
        "duration_ns": int(dur * 1e9),
        "action": {"command": command},
        "output": {"output": "", "returncode": 0},
    }


@pytest.fixture
def recording_path(tmp_path: Path) -> Path:
    """A recording whose tool step carries the real `task_files` command shapes."""
    recording: dict[str, Any] = {
        "format": "mini-swe-agent-recording-1.0",
        "metadata": {"instance_id": "task_files"},
        "events": [
            {"id": 0, "type": "run_start", "timestamp": BASE_TS},
            _model_call(1, 0, BASE_TS, 2.0),
            _tool_call(2, 0, BASE_TS + 2.0, 0.1, MKDIR_COMMAND),
            _tool_call(3, 0, BASE_TS + 2.1, 0.1, HEREDOC_COMMAND),
            _tool_call(4, 0, BASE_TS + 2.2, 0.1, "cat src/main.py"),
            _model_call(5, 1, BASE_TS + 2.5, 1.0),
            {"id": 9, "type": "run_end", "timestamp": BASE_TS + 10},
        ],
    }
    path = tmp_path / "task_files-recording.json"
    path.write_text(json.dumps(recording))
    return path


@pytest.mark.integration
async def test_local_sandbox_executes_a_recorded_command_sequence(
    tmp_path: Path,
) -> None:
    """The exact command shapes the PinchBench task_files trace records."""
    workspace = tmp_path / "ws"
    sandbox = LocalSessionSandbox(workspace=workspace)
    await sandbox.open()
    try:
        await sandbox.run(MKDIR_COMMAND)
        await sandbox.run(HEREDOC_COMMAND)
        listing = await sandbox.run("ls -la src")
        read_back = await sandbox.run("cat src/main.py")
    finally:
        await sandbox.close()

    assert "main.py" in listing.stdout
    assert listing.returncode == 0
    # Byte-exact, not merely "contains": a line-oriented transport that ate the
    # blank lines or swallowed the terminator would still pass a substring check.
    assert (workspace / "src" / "main.py").read_text() == HEREDOC_FILE_TEXT
    assert read_back.stdout == HEREDOC_FILE_TEXT


@pytest.mark.integration
async def test_local_sandbox_reports_a_failing_command_instead_of_raising(
    tmp_path: Path,
) -> None:
    """A nonzero exit is a recorded outcome; only a broken session may raise."""
    sandbox = LocalSessionSandbox(workspace=tmp_path / "ws")
    await sandbox.open()
    try:
        failure = await sandbox.run("cat /definitely/not/here; exit 3")
        recovered = await sandbox.run("echo still-alive")
    finally:
        await sandbox.close()

    assert failure.returncode == 3
    assert not failure.timed_out
    assert failure.duration_s > 0.0
    # The session survives a failure: the next recorded step must still run,
    # and must not read the failed command's output.
    assert recovered.returncode == 0
    assert recovered.stdout.strip() == "still-alive"


@pytest.mark.integration
async def test_local_sandbox_does_not_leak_state_between_commands(
    tmp_path: Path,
) -> None:
    """Each recorded command is a fresh `bash -lc` rooted at the workspace."""
    workspace = tmp_path / "ws"
    sandbox = LocalSessionSandbox(workspace=workspace)
    await sandbox.open()
    try:
        await sandbox.run("mkdir -p sub && cd sub && export MARKER=leaked")
        where = await sandbox.run("pwd")
        marker = await sandbox.run("echo [${MARKER:-unset}]")
    finally:
        await sandbox.close()

    assert Path(where.stdout.strip()).resolve() == workspace.resolve()
    assert marker.stdout.strip() == "[unset]"


@pytest.mark.integration
async def test_lowered_tool_node_runs_through_the_dispatcher(
    recording_path: Path, tmp_path: Path
) -> None:
    """Lowering, dispatch, and sandbox compose: recording in, real files out."""
    graph = from_mini_swe_agent_trace(recording_path, execute_tools=True).graphs[
        "task_files"
    ]
    node = graph.nodes["t0"]
    assert isinstance(node, ToolNode)
    assert node.commands == [MKDIR_COMMAND, HEREDOC_COMMAND, "cat src/main.py"]

    workspace = tmp_path / "ws"
    dispatcher = SandboxToolDispatcher(
        lambda trace_id: LocalSessionSandbox(workspace=workspace / trace_id)
    )
    await dispatcher.open_trace("t-1#0")
    try:
        result = await dispatcher.dispatch(
            node,
            ToolDispatchRequest(node_id="t0"),
            PlacementContext(parent_trace_id="t-1#0", parent_node_id="t0"),
        )
    finally:
        await dispatcher.close_trace("t-1#0")

    assert not result.timed_out
    assert len(result.durations_s) == 3, "one measured duration per recorded command"
    assert all(d > 0.0 for d in result.durations_s)
    # The observation is what the node writes to its channel; the last command's
    # read-back must carry the heredoc body verbatim.
    assert HEREDOC_FILE_TEXT in result.observation
    assert (workspace / "t-1#0" / "src" / "main.py").read_text() == HEREDOC_FILE_TEXT


@pytest.mark.integration
@requires_docker
async def test_docker_sandbox_executes_and_tears_down(tmp_path: Path) -> None:
    sandbox = DockerSessionSandbox(image=IMAGE, workspace=tmp_path / "ws")
    name = sandbox.container_name
    await sandbox.open()
    try:
        result = await sandbox.run("echo containerized")
        inside = await sandbox.run("pwd")
    finally:
        await sandbox.close()

    assert result.stdout.strip() == "containerized"
    assert result.returncode == 0
    # Commands run at the mount point inside the container, not the host path.
    assert inside.stdout.strip() == "/workspace"
    assert await _container_ids(name) == "", "container leaked after close()"


@pytest.mark.integration
@requires_docker
async def test_docker_sandbox_preserves_heredocs_across_the_exec_transport(
    tmp_path: Path,
) -> None:
    """The `docker exec -i` hop is the transport most likely to mangle a heredoc."""
    workspace = tmp_path / "ws"
    sandbox = DockerSessionSandbox(image=IMAGE, workspace=workspace)
    await sandbox.open()
    try:
        await sandbox.run(MKDIR_COMMAND)
        await sandbox.run(HEREDOC_COMMAND)
        read_back = await sandbox.run("cat src/main.py")
    finally:
        await sandbox.close()

    assert read_back.returncode == 0
    assert read_back.stdout == HEREDOC_FILE_TEXT
    # The bind mount is the seam between container work and host inspection.
    assert (workspace / "src" / "main.py").read_text() == HEREDOC_FILE_TEXT


@pytest.mark.integration
@requires_docker
async def test_docker_sandbox_container_is_removed_even_when_a_command_failed(
    tmp_path: Path,
) -> None:
    """Teardown is unconditional; a failed step must not strand a container."""
    sandbox = DockerSessionSandbox(image=IMAGE, workspace=tmp_path / "ws")
    name = sandbox.container_name
    await sandbox.open()
    try:
        failure = await sandbox.run("exit 7")
        assert await _container_ids(name) != "", "container should be up mid-trace"
    finally:
        await sandbox.close()

    assert failure.returncode == 7
    assert await _container_ids(name) == "", "container leaked after close()"


async def _container_ids(name: str) -> str:
    """Ids of any container (running or exited) with this exact name."""
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "ps",
        "-a",
        "-q",
        "-f",
        f"name=^{name}$",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode().strip()
