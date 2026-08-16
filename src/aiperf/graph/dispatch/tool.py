# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ToolNode `_execute` registration for the async-dataflow TraceExecutor.

Deliberately thin, mirroring `_execute_llm`: build a request, await the injected
dispatcher, write the result onto the node's output channel. The policy of what
a tool step DOES lives in the dispatcher, so a different mode is a different
dispatcher rather than an edit here.

A tool step is NOT an endpoint request: it issues no credit, emits no request
record, and contributes to `tool_durations_s` rather than to any latency series.
Nothing reads its output channel in Mode B -- recorded prompts are
self-contained -- but the channel must still be declared and written, because
the channel store rejects a write to an undeclared channel and successors gate
on producer completion.
"""

from __future__ import annotations

from typing import Any

from aiperf.dataset.graph.models import ToolNode
from aiperf.graph.context import NodeExecutionResult, Write, _TraceContext
from aiperf.graph.executor import TraceExecutor
from aiperf.graph.placement import PlacementContext
from aiperf.graph.tool_dispatch.protocols import ToolDispatchRequest

__all__ = ["_execute_tool"]


async def _execute_tool(
    self: TraceExecutor,
    node: ToolNode,
    node_id: str,
    inputs: dict[str, Any],
    ctx: _TraceContext,
) -> NodeExecutionResult:
    """Dispatch one ToolNode through the injected tool dispatcher."""
    if self._tool_dispatcher is None:
        raise RuntimeError(
            f"node {node_id!r} is a ToolNode but the executor has no tool "
            "dispatcher; tool execution requires one"
        )

    result = await self._tool_dispatcher.dispatch(
        node,
        ToolDispatchRequest(node_id=node_id),
        PlacementContext(parent_trace_id=ctx.trace.id, parent_node_id=node_id),
    )
    ctx.tool_durations_s.extend(result.durations_s)
    return NodeExecutionResult(
        writes=[Write(channel=node.output, value=result.observation)]
    )


TraceExecutor.__dict__["_execute"].register(ToolNode, _execute_tool)
