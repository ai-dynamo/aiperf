# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LlmNode `_execute` registration for the async-dataflow TraceExecutor.

Build a `DispatchRequest` (node id), await `credit_issuer.dispatch(node,
request, placement_ctx, first_token_cb=...)`, and write the returned value onto
`node.output` (a type-correct empty list for messages-typed channels, the
placeholder string otherwise). The `first_token_cb` is the per-dispatch stamp
closure (`_make_first_token_stamp`) the adapter fires when this node's
`FirstToken` arrives, releasing any first-token-anchored successor (post-TTFT
anchoring).

Errors propagate to the executor: `GraphDispatchError` is contained
mid-conversation by `_handle_node_exception` (segment-store parses write a
type-correct sentinel to the node's channels so the conversation continues;
issuer refusals and stickiness errors re-raise as trace-stops). Anything else
(`asyncio.TimeoutError`, transport errors) unwinds the trace: `node.output`
stays UNSET and the `mark_producer_done` in `_fire`'s `finally` orphans
waiters whose count target is unreachable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aiperf.dataset.graph.models import (
    ChannelType,
    LlmNode,
)
from aiperf.graph.context import (
    NodeExecutionResult,
    Write,
    _TraceContext,
)
from aiperf.graph.executor import TraceExecutor
from aiperf.graph.placement import (
    DispatchRequest,
    PlacementContext,
)

__all__ = ["_execute_llm"]


def _make_first_token_stamp(
    ctx: _TraceContext, node_id: str, loop_wall_us: Callable[[], float]
) -> Callable[[], None]:
    """Build the zero-arg release hook the PRODUCING node registers on its own
    dispatch, releasing any first-token-anchored successor.

    Passed as the per-dispatch ``first_token_cb`` kwarg; the dispatch adapter
    parks it and invokes it AT MOST ONCE when this node's ``FirstToken`` arrives.
    The stamp records the observing loop's wall time and sets the node's
    first-token latch so a successor gated on this node's observed first token is
    released. A late / duplicate invocation is a guarded no-op: the wall is never
    overwritten and the clock is not re-read (the guard returns before the read).
    """

    def _stamp() -> None:
        if node_id in ctx.node_first_token_wall_us:
            return  # late / duplicate first token: no-op
        ctx.node_first_token_wall_us[node_id] = loop_wall_us()
        ctx.first_token_event(node_id).set()

    return _stamp


async def _execute_llm(
    self: TraceExecutor,
    node: LlmNode,
    node_id: str,
    inputs: dict[str, Any],
    ctx: _TraceContext,
) -> NodeExecutionResult:
    """Dispatch one LlmNode through the injected credit issuer."""
    spec = self._parsed.graph.state.get(node.output)
    channel_type = spec.type if spec is not None else ChannelType.TEXT

    # The scheduler already knows which id it is firing. Recovering it by
    # identity-scanning the node dict is O(N) per dispatch and, worse, silently
    # reports the FIRST id when the same frozen struct is reused under several
    # ids (msgspec structs are interned freely by adapters).
    producer_id = node_id

    request = DispatchRequest(
        node_id=producer_id,
    )

    placement_ctx = PlacementContext(
        parent_trace_id=ctx.trace.id,
        parent_node_id=producer_id,
    )

    response = await self._credit_issuer.dispatch(
        node,
        request,
        placement_ctx,
        first_token_cb=_make_first_token_stamp(ctx, producer_id, self._loop_wall_us),
    )

    # The credit path resolves a content-free placeholder string (content stays
    # worker-side). A messages-typed output channel needs a type-correct empty
    # instead: the channel's reducer runs over whatever is written, and
    # `add_messages` rejects non-list values.
    value: Any = [] if channel_type is ChannelType.MESSAGES else response
    writes: list[Write] = [Write(channel=node.output, value=value)]
    return NodeExecutionResult(writes=writes)


TraceExecutor.__dict__["_execute"].register(LlmNode, _execute_llm)
