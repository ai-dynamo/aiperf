# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-level sandbox lifecycle owners.

A ``SandboxProvider`` handles everything that happens OUTSIDE the per-trace
timing window:

* ``setup()`` — called by ``AgentGraphReplayStrategy.setup_phase()`` BEFORE
  the phase's baseline boundary is captured, so pre-pull and container
  startup time are never counted against benchmark throughput.
* ``make_sandbox()`` — called per trace instance to produce the sandbox that
  ``SandboxToolDispatcher`` will open/run/close.
* ``teardown()`` — called from ``teardown_phase()`` for phase-level cleanup.

Three concrete implementations:

* ``LocalSandboxProvider`` — no-op setup; each trace runs in a local shell.
* ``DockerSandboxProvider`` (no pool) — pre-pulls images in ``setup()``,
  creates one ``DockerSessionSandbox`` per trace instance.  Container startup
  cost lands in ``sandbox_setup_s`` (inside the trace timing window).
* ``DockerSandboxProvider`` (with pool) — additionally pre-starts
  ``pool_size`` containers per image in ``setup()``.  ``make_sandbox()``
  returns a ``PooledDockerSandbox`` whose ``open()`` is an async checkout
  (blocks until a slot is free, no docker run), so ``sandbox_setup_s → 0``.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.dataset.graph.models import TraceRecord
from aiperf.graph.sandbox.docker import CONTAINER_WORKSPACE, DockerSessionSandbox
from aiperf.graph.sandbox.local import DEFAULT_INTERPRETER, LocalSessionSandbox
from aiperf.graph.sandbox.pool import ContainerPool, PooledDockerSandbox
from aiperf.graph.sandbox.protocols import ToolSandbox

__all__ = [
    "SandboxProvider",
    "LocalSandboxProvider",
    "DockerSandboxProvider",
]

_logger = AIPerfLogger(__name__)

# SWE-Bench images are initialized with a task checkout at /testbed. Reusing a
# container would reuse the mutated checkout instead of starting the next trace
# from the image's pristine task state.
_NON_POOLED_CWDS = frozenset({"/testbed"})


@runtime_checkable
class SandboxProvider(Protocol):
    """Phase-level lifecycle owner for tool sandbox resources."""

    async def setup(self) -> None:
        """Pre-phase setup — pre-pull images, start pools, etc.

        Called once before the first trace of a phase starts, outside the
        timing window. A ``RuntimeError`` raised here fails the run before
        any measurement begins.
        """
        ...

    def make_sandbox(self, instance_id: str, trace: TraceRecord) -> ToolSandbox:
        """Return a fresh sandbox for one trace instance.

        Called once per running instance from ``_build_tool_dispatcher``.
        The sandbox's own ``open()`` / ``run()`` / ``close()`` are called
        by ``SandboxToolDispatcher`` inside the timing window.
        """
        ...

    async def teardown(self) -> None:
        """Post-phase cleanup. Called from ``teardown_phase()``."""
        ...


def _instance_slug(instance_id: str) -> str:
    """Filesystem-safe slug from an instance id (strips ``::`` and path separators)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", instance_id)


def _image_slug(image: str) -> str:
    """Filesystem-safe name for an image (used as pool workspace subdirectory)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", image)[:64]


class LocalSandboxProvider:
    """Run recorded commands in a local shell, one per trace instance."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root

    async def setup(self) -> None:
        pass

    def make_sandbox(self, instance_id: str, trace: TraceRecord) -> ToolSandbox:
        del trace
        return LocalSessionSandbox(
            workspace=self._workspace_root / _instance_slug(instance_id)
        )

    async def teardown(self) -> None:
        pass


class DockerSandboxProvider:
    """Run recorded commands inside per-trace or pooled Docker containers.

    Two modes selected at construction time:

    **No pool** (``pool_size=None``, default):
        ``setup()`` pre-pulls all unique images concurrently so no trace
        ever blocks on an image download.  Each ``make_sandbox()`` returns
        a ``DockerSessionSandbox`` whose ``open()`` starts a fresh container
        — startup cost lands in ``sandbox_setup_s``.

    **Pooled** (``pool_size=N``, requires ``persistent_session=False``):
        ``setup()`` pre-pulls images AND starts ``N`` containers per image
        concurrently, all before the timing baseline is captured.  Each
        ``make_sandbox()`` returns a ``PooledDockerSandbox`` whose ``open()``
        does an async checkout from the pool (no docker run), so
        ``sandbox_setup_s → 0``.  Slots are returned and their workspaces
        cleared on ``close()``.  Pool is torn down in ``teardown()``.
    """

    def __init__(
        self,
        images: frozenset[str],
        workspace_root: Path,
        *,
        global_image: str | None = None,
        persistent_session: bool = False,
        pool_size: int | None = None,
        non_pooled_images: frozenset[str] = frozenset(),
    ) -> None:
        self._images = images
        self._workspace_root = workspace_root
        self._global_image = global_image
        self._persistent_session = persistent_session
        # Pool is only used for fresh-exec mode; persistent sessions keep
        # a bash open inside the container which would be shared across
        # concurrent checkouts and is not safe to reuse.
        self._pool_size = pool_size if not persistent_session else None
        self._non_pooled_images = non_pooled_images
        self._pools: dict[str, ContainerPool] = {}

    async def setup(self) -> None:
        # Step 1: pre-pull all images concurrently.
        if self._images:
            _logger.info(
                lambda: f"pre-pulling {len(self._images)} tool image(s): "
                f"{sorted(self._images)}"
            )
            await asyncio.gather(*[self._pull(img) for img in self._images])

        # Step 2: if pooling is requested, start containers now while we're
        # still outside the timing window.
        if self._pool_size is not None and self._images:
            pool_workspace = self._workspace_root / "pool"
            pools = {
                img: ContainerPool(
                    image=img,
                    pool_size=self._pool_size,
                    workspace_root=pool_workspace / _image_slug(img),
                )
                for img in self._images
                if img not in self._non_pooled_images
            }
            await asyncio.gather(*[pool.start() for pool in pools.values()])
            self._pools = pools

    async def _pull(self, image: str) -> None:
        # Skip the registry pull when the image already exists locally — local
        # builds (e.g. `docker build -t my-task:latest .`) are never in a
        # registry, so `docker pull` would fail with "access denied" even
        # though the image is perfectly usable.
        inspect = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await inspect.communicate()
        if inspect.returncode == 0:
            _logger.info(
                lambda image=image: f"image already local, skipping pull: {image!r}"
            )
            return

        proc = await asyncio.create_subprocess_exec(
            "docker",
            "pull",
            image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to pre-pull tool image {image!r}: "
                f"{stdout.decode(errors='replace').strip()}"
            )
        _logger.info(lambda image=image: f"pre-pull complete: {image!r}")

    def make_sandbox(self, instance_id: str, trace: TraceRecord) -> ToolSandbox:
        image = (
            trace.tool_sandbox.container if trace.tool_sandbox else None
        ) or self._global_image
        slug = _instance_slug(instance_id)
        cwd = trace.tool_sandbox.cwd if trace.tool_sandbox else None
        interpreter = trace.tool_sandbox.interpreter if trace.tool_sandbox else None
        if image is None:
            return LocalSessionSandbox(
                workspace=self._workspace_root / slug,
                interpreter=interpreter or DEFAULT_INTERPRETER,
            )
        if image in self._pools and cwd not in _NON_POOLED_CWDS:
            return PooledDockerSandbox(
                self._pools[image],
                cwd=cwd or CONTAINER_WORKSPACE,
                interpreter=interpreter or DEFAULT_INTERPRETER,
            )
        return DockerSessionSandbox(
            image=image,
            workspace=self._workspace_root / slug,
            container_name=f"aiperf-tool-{slug[-48:]}",
            cwd=cwd or CONTAINER_WORKSPACE,
            interpreter=interpreter or DEFAULT_INTERPRETER,
            persistent_session=self._persistent_session,
        )

    async def teardown(self) -> None:
        if self._pools:
            await asyncio.gather(
                *[pool.stop() for pool in self._pools.values()],
                return_exceptions=True,
            )
            self._pools.clear()
