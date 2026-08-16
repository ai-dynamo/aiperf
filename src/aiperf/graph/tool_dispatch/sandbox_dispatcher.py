# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The default `ToolDispatcher`: run a step's recorded commands in a sandbox.

Owns the per-trace sandbox lifecycle so the executor does not have to. The
sandbox itself is injected as a factory, keeping *where commands run* (local
shell vs container) independent of *what a tool step is*.
"""

from __future__ import annotations

from collections.abc import Callable

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.dataset.graph.models import ToolNode
from aiperf.graph.placement import PlacementContext
from aiperf.graph.sandbox.protocols import ToolSandbox
from aiperf.graph.tool_dispatch.protocols import (
    ToolDispatchRequest,
    ToolDispatchResult,
)

__all__ = ["SandboxToolDispatcher"]

_logger = AIPerfLogger(__name__)


class SandboxToolDispatcher:
    """Execute each recorded command of a step in this trace's sandbox.

    ONE INSTANCE PER TRACE; not reusable across concurrent traces. It holds a
    single current sandbox, so `open_trace` for a second trace would run that
    trace's commands in the first one's sandbox and `close_trace` would tear it
    down out from under the first. The `trace_id`-keyed method signatures name
    the seam's contract (`ToolDispatcher` must serve any keying a future mode
    needs), not this implementation's capacity -- so `open_trace` asserts rather
    than silently rebinding.
    """

    def __init__(self, sandbox_factory: Callable[[str], ToolSandbox]) -> None:
        self._sandbox_factory = sandbox_factory
        self._sandbox: ToolSandbox | None = None
        self._trace_id: str | None = None

    async def open_trace(self, trace_id: str) -> None:
        if self._sandbox is not None:
            raise RuntimeError(
                f"trace {trace_id!r}: this SandboxToolDispatcher already holds an "
                f"open sandbox (for trace {self._trace_id!r}). One instance serves "
                "one trace; build a new dispatcher per trace instead of sharing "
                "one across concurrent traces, which would run this trace's "
                "commands in the other's sandbox and close it out from under "
                "that trace."
            )
        self._sandbox = self._sandbox_factory(trace_id)
        self._trace_id = trace_id
        await self._sandbox.open()

    async def dispatch(
        self,
        node: ToolNode,
        request: ToolDispatchRequest,
        ctx: PlacementContext,
    ) -> ToolDispatchResult:
        if self._sandbox is None:
            raise RuntimeError(
                f"node {request.node_id!r}: sandbox dispatcher used before "
                "open_trace; the executor must bracket the trace"
            )

        observations: list[str] = []
        durations: list[float] = []
        timed_out = False
        for command in node.commands:
            result = await self._sandbox.run(command, timeout_s=node.timeout_s)
            durations.append(result.duration_s)
            observations.append(result.stdout)
            if result.timed_out:
                timed_out = True
                _logger.warning(
                    lambda cmd=command: f"node {request.node_id!r}: tool command "
                    f"timed out: {cmd[:80]!r}"
                )
                break
        return ToolDispatchResult(
            observation="\n".join(observations),
            durations_s=durations,
            timed_out=timed_out,
        )

    async def close_trace(self, trace_id: str) -> None:
        # Symmetric with `open_trace`'s one-instance assertion: closing under a
        # DIFFERENT id than the one opened means the caller lost track of which
        # dispatcher belongs to which trace, and silently tearing down the
        # wrong sandbox is the exact failure `open_trace` refuses to allow.
        if self._sandbox is not None and trace_id != self._trace_id:
            raise RuntimeError(
                f"trace {trace_id!r}: this SandboxToolDispatcher holds the "
                f"sandbox for trace {self._trace_id!r}. Closing it here would "
                "tear down another trace's sandbox mid-run; build one "
                "dispatcher per trace."
            )
        sandbox, self._sandbox = self._sandbox, None
        self._trace_id = None
        if sandbox is not None:
            await sandbox.close()
