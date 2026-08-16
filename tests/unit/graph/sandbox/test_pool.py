# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ContainerPool and PooledDockerSandbox."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aiperf.graph.sandbox import pool as pool_module
from aiperf.graph.sandbox.pool import (
    ContainerPool,
    PooledDockerSandbox,
    PoolSlot,
    _clear_workspace,
)

# ---------------------------------------------------------------------------
# ContainerPool — start/checkout/return/stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_start_creates_slots_and_makes_them_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []

    class _OKProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec(*argv: str, **_kw: object) -> _OKProc:
        if "run" in argv:
            container_name = argv[argv.index("--name") + 1]
            started.append(container_name)
        return _OKProc()

    monkeypatch.setattr(pool_module.asyncio, "create_subprocess_exec", _fake_exec)

    pool = ContainerPool(image="img:1", pool_size=3, workspace_root=tmp_path)
    await pool.start()

    assert len(started) == 3
    assert pool._slots.qsize() == 3


@pytest.mark.asyncio
async def test_pool_checkout_blocks_until_slot_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _OKProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec(*_a: object, **_k: object) -> _OKProc:
        return _OKProc()

    monkeypatch.setattr(pool_module.asyncio, "create_subprocess_exec", _fake_exec)

    pool = ContainerPool(image="img:1", pool_size=1, workspace_root=tmp_path)
    await pool.start()

    slot1 = await pool.checkout()
    assert pool._slots.qsize() == 0

    # Second checkout should block; resolve it by returning slot1 concurrently.
    async def _return_after_tick() -> None:
        await asyncio.sleep(0)
        await pool.return_slot(slot1)

    asyncio.ensure_future(_return_after_tick())
    slot2 = await asyncio.wait_for(pool.checkout(), timeout=1.0)
    assert slot2 is slot1  # same slot reused


@pytest.mark.asyncio
async def test_pool_return_clears_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _OKProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec(*_a: object, **_k: object) -> _OKProc:
        return _OKProc()

    monkeypatch.setattr(pool_module.asyncio, "create_subprocess_exec", _fake_exec)

    pool = ContainerPool(image="img:1", pool_size=1, workspace_root=tmp_path)
    await pool.start()
    slot = await pool.checkout()

    # Write something into the slot workspace.
    (slot.workspace / "leftover.txt").write_text("junk")

    await pool.return_slot(slot)

    # Workspace must be empty after return.
    assert not any(slot.workspace.iterdir())


@pytest.mark.asyncio
async def test_pool_stop_runs_docker_rm_for_all_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    removed: list[str] = []

    class _OKProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

        async def wait(self) -> int:
            return 0

    async def _fake_exec(*argv: str, **_kw: object) -> _OKProc:
        if "rm" in argv and "-f" in argv:
            removed.append(argv[-1])
        return _OKProc()

    monkeypatch.setattr(pool_module.asyncio, "create_subprocess_exec", _fake_exec)

    pool = ContainerPool(image="img:1", pool_size=2, workspace_root=tmp_path)
    await pool.start()
    await pool.stop()

    assert len(removed) == 2


# ---------------------------------------------------------------------------
# PooledDockerSandbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pooled_sandbox_open_checks_out_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _OKProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec(*_a: object, **_k: object) -> _OKProc:
        return _OKProc()

    monkeypatch.setattr(pool_module.asyncio, "create_subprocess_exec", _fake_exec)

    pool = ContainerPool(image="img:1", pool_size=1, workspace_root=tmp_path)
    await pool.start()

    sandbox = PooledDockerSandbox(pool)
    assert sandbox._slot is None
    await sandbox.open()
    assert sandbox._slot is not None
    assert pool._slots.qsize() == 0  # slot checked out


@pytest.mark.asyncio
async def test_pooled_sandbox_run_sends_fresh_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_argv: list[list[str]] = []

    class _DoneProc:
        pid = -1
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"output\n", b""

    async def _fake_exec(*argv: str, **_kw: object) -> _DoneProc:
        seen_argv.append(list(argv))
        return _DoneProc()

    monkeypatch.setattr(pool_module.asyncio, "create_subprocess_exec", _fake_exec)

    pool = ContainerPool(image="img:1", pool_size=1, workspace_root=tmp_path)
    # Manually inject a slot (bypass docker run).
    slot = PoolSlot(
        image="img:1", container_name="aiperf-pool-test-0", workspace=tmp_path
    )
    pool._all_slots = [slot]
    pool._slots.put_nowait(slot)

    sandbox = PooledDockerSandbox(pool)
    await sandbox.open()
    result = await sandbox.run("echo hi")

    assert result.stdout == "output\n"
    exec_calls = [a for a in seen_argv if a[:2] == ["docker", "exec"]]
    assert len(exec_calls) == 1
    assert "-w" in exec_calls[0]
    assert "aiperf-pool-test-0" in exec_calls[0]
    assert "echo hi" in exec_calls[0]
    assert "-i" not in exec_calls[0]  # no persistent session flag


@pytest.mark.asyncio
async def test_pooled_sandbox_run_uses_trace_shell_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_argv: list[list[str]] = []

    class _DoneProc:
        pid = -1
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec(*argv: str, **_kw: object) -> _DoneProc:
        seen_argv.append(list(argv))
        return _DoneProc()

    monkeypatch.setattr(pool_module.asyncio, "create_subprocess_exec", _fake_exec)
    pool = ContainerPool(image="img:1", pool_size=1, workspace_root=tmp_path)
    slot = PoolSlot(
        image="img:1", container_name="aiperf-pool-test-0", workspace=tmp_path
    )
    pool._all_slots = [slot]
    pool._slots.put_nowait(slot)
    sandbox = PooledDockerSandbox(pool, cwd="/testbed", interpreter=("bash", "-c"))

    await sandbox.open()
    await sandbox.run("echo hi")

    assert seen_argv == [
        [
            "docker",
            "exec",
            "-w",
            "/testbed",
            "aiperf-pool-test-0",
            "bash",
            "-c",
            "echo hi",
        ]
    ]


@pytest.mark.asyncio
async def test_pooled_sandbox_timeout_replaces_slot_before_return(
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

    class _DockerLifecycleProc:
        pid = -1
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

        async def wait(self) -> int:
            return 0

    async def _fake_exec(*argv: str, **_kwargs: object) -> object:
        calls.append(list(argv))
        if argv[:2] == ("docker", "exec"):
            return _TimedOutExec()
        return _DockerLifecycleProc()

    monkeypatch.setattr(pool_module.asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(
        pool_module.LocalSessionSandbox,
        "_kill_process_group",
        staticmethod(lambda proc: None),
    )
    pool = ContainerPool(image="img:1", pool_size=1, workspace_root=tmp_path)
    old_slot = PoolSlot(
        image="img:1",
        container_name="aiperf-pool-old",
        workspace=tmp_path / "old",
    )
    pool._all_slots = [old_slot]
    pool._slots.put_nowait(old_slot)
    sandbox = PooledDockerSandbox(pool)
    await sandbox.open()

    result = await sandbox.run("sleep forever")

    replacement = await asyncio.wait_for(pool.checkout(), timeout=1.0)
    assert result.timed_out is True
    assert replacement.container_name != old_slot.container_name
    assert replacement.workspace != old_slot.workspace
    assert ["docker", "rm", "-f", old_slot.container_name] in calls
    with pytest.raises(RuntimeError, match=r"open\(\) must be called"):
        await sandbox.run("must not run")


@pytest.mark.asyncio
async def test_pooled_sandbox_close_returns_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _OKProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec2(*_a: object, **_k: object) -> _OKProc:
        return _OKProc()

    monkeypatch.setattr(
        pool_module.asyncio,
        "create_subprocess_exec",
        _fake_exec2,
    )

    pool = ContainerPool(image="img:1", pool_size=1, workspace_root=tmp_path)
    slot = PoolSlot(
        image="img:1", container_name="aiperf-pool-test-0", workspace=tmp_path
    )
    pool._all_slots = [slot]
    pool._slots.put_nowait(slot)

    sandbox = PooledDockerSandbox(pool)
    await sandbox.open()
    assert pool._slots.qsize() == 0
    await sandbox.close()
    assert pool._slots.qsize() == 1
    assert sandbox._slot is None


# ---------------------------------------------------------------------------
# _clear_workspace
# ---------------------------------------------------------------------------


def test_clear_workspace_removes_files_and_dirs(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("data")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.py").write_text("code")

    _clear_workspace(tmp_path)

    assert not list(tmp_path.iterdir())


def test_clear_workspace_noop_on_missing_dir(tmp_path: Path) -> None:
    _clear_workspace(tmp_path / "nonexistent")  # must not raise
