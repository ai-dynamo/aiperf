# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SandboxProvider implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.dataset.graph.models import ToolSandboxSpec, TraceRecord
from aiperf.graph.sandbox import provider as provider_module
from aiperf.graph.sandbox.docker import DockerSessionSandbox
from aiperf.graph.sandbox.local import LocalSessionSandbox
from aiperf.graph.sandbox.provider import DockerSandboxProvider, LocalSandboxProvider


@pytest.mark.asyncio
async def test_local_provider_setup_is_noop(tmp_path: Path) -> None:
    p = LocalSandboxProvider(workspace_root=tmp_path)
    await p.setup()  # must not raise or block


def test_local_provider_makes_local_sandbox(tmp_path: Path) -> None:
    p = LocalSandboxProvider(workspace_root=tmp_path)
    sandbox = p.make_sandbox("t-1::abc", TraceRecord(id="t-1"))
    assert isinstance(sandbox, LocalSessionSandbox)
    assert sandbox._workspace == tmp_path / "t-1-abc"


def test_local_provider_slug_strips_unsafe_chars(tmp_path: Path) -> None:
    p = LocalSandboxProvider(workspace_root=tmp_path)
    sandbox = p.make_sandbox("trace/id::nonce", TraceRecord(id="x"))
    # slashes and colons become dashes
    assert "/" not in sandbox._workspace.name
    assert ":" not in sandbox._workspace.name


@pytest.mark.asyncio
async def test_docker_provider_setup_calls_pull_for_each_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pulled: list[str] = []

    async def _fake_pull(self: DockerSandboxProvider, image: str) -> None:
        pulled.append(image)

    monkeypatch.setattr(DockerSandboxProvider, "_pull", _fake_pull)

    p = DockerSandboxProvider(
        images=frozenset({"img-a:1", "img-b:2"}),
        workspace_root=tmp_path,
    )
    await p.setup()

    assert sorted(pulled) == ["img-a:1", "img-b:2"]


@pytest.mark.asyncio
async def test_docker_provider_setup_skips_pull_when_no_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pulled: list[str] = []

    async def _fake_pull(self: DockerSandboxProvider, image: str) -> None:
        pulled.append(image)

    monkeypatch.setattr(DockerSandboxProvider, "_pull", _fake_pull)

    p = DockerSandboxProvider(images=frozenset(), workspace_root=tmp_path)
    await p.setup()

    assert pulled == []


def test_docker_provider_makes_docker_sandbox_with_global_image(
    tmp_path: Path,
) -> None:
    p = DockerSandboxProvider(
        images=frozenset({"task:latest"}),
        workspace_root=tmp_path,
        global_image="task:latest",
    )
    sandbox = p.make_sandbox("t-1::abc", TraceRecord(id="t-1"))
    assert isinstance(sandbox, DockerSessionSandbox)
    assert "task:latest" in sandbox.start_argv()
    assert sandbox._workspace == tmp_path / "t-1-abc"


def test_docker_provider_per_trace_image_overrides_global(tmp_path: Path) -> None:
    p = DockerSandboxProvider(
        images=frozenset({"global:latest", "per-trace:v1"}),
        workspace_root=tmp_path,
        global_image="global:latest",
    )
    trace = TraceRecord(
        id="t-1", tool_sandbox=ToolSandboxSpec(container="per-trace:v1")
    )
    sandbox = p.make_sandbox("t-1::abc", trace)
    assert "per-trace:v1" in sandbox.start_argv()
    assert "global:latest" not in sandbox.start_argv()


def test_docker_provider_uses_local_sandbox_when_trace_has_no_image(
    tmp_path: Path,
) -> None:
    p = DockerSandboxProvider(
        images=frozenset({"task:latest"}),
        workspace_root=tmp_path,
        global_image=None,
    )

    sandbox = p.make_sandbox("local::abc", TraceRecord(id="local"))

    assert isinstance(sandbox, LocalSessionSandbox)
    assert sandbox._workspace == tmp_path / "local-abc"


def test_docker_provider_forwards_persistent_session_flag(tmp_path: Path) -> None:
    p = DockerSandboxProvider(
        images=frozenset({"img:1"}),
        workspace_root=tmp_path,
        global_image="img:1",
        persistent_session=True,
    )
    sandbox = p.make_sandbox("t-1::abc", TraceRecord(id="t-1"))
    assert isinstance(sandbox, DockerSessionSandbox)
    assert sandbox._persistent_session is True


@pytest.mark.asyncio
async def test_docker_provider_pull_raises_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailProc:
        returncode = 1

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"pull access denied", b""

    async def _fake_exec(*_args: object, **_kwargs: object) -> _FailProc:
        return _FailProc()

    monkeypatch.setattr(provider_module.asyncio, "create_subprocess_exec", _fake_exec)

    p = DockerSandboxProvider(
        images=frozenset({"bad:image"}),
        workspace_root=tmp_path,
        global_image="bad:image",
    )
    with pytest.raises(RuntimeError, match="failed to pre-pull"):
        await p.setup()


@pytest.mark.asyncio
async def test_docker_provider_pull_skips_when_image_exists_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `docker image inspect` succeeds, no `docker pull` is issued."""
    calls: list[list[str]] = []

    class _OKProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"sha256:abc123", b""

    async def _fake_exec(*argv: str, **_kw: object) -> _OKProc:
        calls.append(list(argv))
        return _OKProc()

    monkeypatch.setattr(provider_module.asyncio, "create_subprocess_exec", _fake_exec)

    p = DockerSandboxProvider(
        images=frozenset({"local:img"}),
        workspace_root=tmp_path,
        global_image="local:img",
    )
    await p.setup()

    # Only the inspect call should have been made — no pull.
    assert any("inspect" in c for c in calls)
    assert not any("pull" in c for c in calls)


# ---------------------------------------------------------------------------
# DockerSandboxProvider — pooled mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_docker_provider_with_pool_size_starts_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pool_size=N triggers ContainerPool.start() in addition to image pull."""
    pulled: list[str] = []
    pool_started: list[int] = []

    async def _fake_pull(self: DockerSandboxProvider, image: str) -> None:
        pulled.append(image)

    async def _fake_pool_start(self_pool: object) -> None:
        pool_started.append(1)

    monkeypatch.setattr(DockerSandboxProvider, "_pull", _fake_pull)
    # Patch ContainerPool.start so we don't need a Docker daemon.
    from aiperf.graph.sandbox.pool import ContainerPool

    monkeypatch.setattr(ContainerPool, "start", _fake_pool_start)

    p = DockerSandboxProvider(
        images=frozenset({"img:1"}),
        workspace_root=tmp_path,
        global_image="img:1",
        pool_size=4,
    )
    await p.setup()

    assert pulled == ["img:1"]
    assert pool_started == [1]
    assert "img:1" in p._pools


@pytest.mark.asyncio
async def test_docker_provider_make_sandbox_returns_pooled_when_pool_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiperf.graph.sandbox.pool import ContainerPool, PooledDockerSandbox, PoolSlot

    async def _fake_pull(self: DockerSandboxProvider, image: str) -> None:
        pass

    async def _fake_pool_start(self_pool: ContainerPool) -> None:
        slot = PoolSlot(
            image="img:1",
            container_name="aiperf-pool-fake-0",
            workspace=tmp_path / "slot-0",
        )
        self_pool._all_slots = [slot]
        self_pool._slots.put_nowait(slot)

    monkeypatch.setattr(DockerSandboxProvider, "_pull", _fake_pull)
    monkeypatch.setattr(ContainerPool, "start", _fake_pool_start)

    p = DockerSandboxProvider(
        images=frozenset({"img:1"}),
        workspace_root=tmp_path,
        global_image="img:1",
        pool_size=1,
    )
    await p.setup()

    sandbox = p.make_sandbox("t-1::abc", TraceRecord(id="t-1"))
    assert isinstance(sandbox, PooledDockerSandbox)


@pytest.mark.asyncio
async def test_docker_provider_swebench_trace_bypasses_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiperf.graph.sandbox.pool import ContainerPool

    async def _fake_pull(self: DockerSandboxProvider, image: str) -> None:
        pass

    async def _fake_pool_start(self_pool: ContainerPool) -> None:
        pass

    monkeypatch.setattr(DockerSandboxProvider, "_pull", _fake_pull)
    monkeypatch.setattr(ContainerPool, "start", _fake_pool_start)
    provider = DockerSandboxProvider(
        images=frozenset({"swebench:latest"}),
        workspace_root=tmp_path,
        pool_size=1,
    )
    await provider.setup()
    trace = TraceRecord(
        id="swebench",
        tool_sandbox=ToolSandboxSpec(
            container="swebench:latest",
            cwd="/testbed",
            interpreter=("bash", "-c"),
        ),
    )

    sandbox = provider.make_sandbox("swebench::abc", trace)

    assert isinstance(sandbox, DockerSessionSandbox)
    assert sandbox._workspace_in_sandbox == "/testbed"


def test_docker_provider_persistent_session_disables_pool(tmp_path: Path) -> None:
    """persistent_session=True must force pool_size to None (pool not safe to share)."""
    p = DockerSandboxProvider(
        images=frozenset({"img:1"}),
        workspace_root=tmp_path,
        global_image="img:1",
        persistent_session=True,
        pool_size=4,
    )
    # Provider must internally suppress the pool.
    assert p._pool_size is None


@pytest.mark.asyncio
async def test_docker_provider_does_not_start_pool_for_swebench_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiperf.graph.sandbox.pool import ContainerPool

    started: list[str] = []

    async def _fake_pull(self: DockerSandboxProvider, image: str) -> None:
        pass

    async def _fake_pool_start(self_pool: ContainerPool) -> None:
        started.append(self_pool._image)

    monkeypatch.setattr(DockerSandboxProvider, "_pull", _fake_pull)
    monkeypatch.setattr(ContainerPool, "start", _fake_pool_start)
    provider = DockerSandboxProvider(
        images=frozenset({"swebench:latest"}),
        workspace_root=tmp_path,
        pool_size=2,
        non_pooled_images=frozenset({"swebench:latest"}),
    )

    await provider.setup()

    assert started == []
    assert provider._pools == {}


@pytest.mark.asyncio
async def test_docker_provider_teardown_stops_pools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aiperf.graph.sandbox.pool import ContainerPool

    stopped: list[int] = []

    async def _fake_pool_stop(self_pool: ContainerPool) -> None:
        stopped.append(1)

    async def _fake_pull(self: DockerSandboxProvider, image: str) -> None:
        pass

    async def _fake_pool_start(self_pool: ContainerPool) -> None:
        pass

    monkeypatch.setattr(DockerSandboxProvider, "_pull", _fake_pull)
    monkeypatch.setattr(ContainerPool, "start", _fake_pool_start)
    monkeypatch.setattr(ContainerPool, "stop", _fake_pool_stop)

    p = DockerSandboxProvider(
        images=frozenset({"img:1"}),
        workspace_root=tmp_path,
        global_image="img:1",
        pool_size=2,
    )
    await p.setup()
    await p.teardown()

    assert stopped == [1]
    assert p._pools == {}
