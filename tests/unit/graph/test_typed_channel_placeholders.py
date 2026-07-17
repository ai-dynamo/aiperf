# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Type-correct channel placeholders on the LlmNode credit path.

The credit path resolves a content-free placeholder string (`""`) and the
failure path writes a gate sentinel; both flow into the channel store, whose
global `snapshot_at_seq` reduces EVERY channel. A messages-typed output channel
therefore needs a type-correct empty (`[]`) — `add_messages` rejects non-list
values — on both the success path (`dispatch/llm.py`) and the dispatch-failure
sentinel branch (`executor._handle_node_exception`). These tests run a two-node
chain where node `a` writes a messages-typed channel and node `b`'s snapshot
reduces it: with untyped placeholders either path raises `TypeError` and the
trace errors.
"""

import asyncio
from typing import Any

import pytest

from aiperf.dataset.graph.models import (
    ChannelRequirement,
    ChannelSpec,
    ChannelType,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ReducerName,
    StaticEdge,
    TraceRecord,
)
from aiperf.dataset.graph.segment_ir.pool import SegmentPool
from aiperf.graph.credit_dispatch_adapter import GraphDispatchError
from aiperf.graph.executor import TraceExecutor


def _messages_chain_parsed(*, with_pool: bool) -> ParsedGraph:
    graph = GraphRecord(
        state={
            "a_out": ChannelSpec(
                type=ChannelType.MESSAGES, reducer=ReducerName.ADD_MESSAGES
            ),
            "b_out": ChannelSpec(),
        },
        nodes={
            "a": LlmNode(
                prompt=[{"role": "user", "content": "q1"}],
                output="a_out",
            ),
            "b": LlmNode(
                prompt=[{"role": "user", "content": "q2"}],
                output="b_out",
                inputs=[ChannelRequirement(channel="a_out", count=1)],
            ),
        },
        edges=[
            StaticEdge(source="START", target="a"),
            StaticEdge(source="a", target="b"),
            StaticEdge(source="b", target="END"),
        ],
    )
    return ParsedGraph(
        graph=graph,
        traces=[TraceRecord(id="t1")],
        segment_pool=SegmentPool() if with_pool else None,
    )


class _EchoIssuer:
    """Resolves every dispatch with the credit path's placeholder string."""

    async def dispatch(
        self, node: Any, request: Any, ctx: Any, first_token_cb: Any = None
    ) -> str:
        return ""


class _FailFirstIssuer(_EchoIssuer):
    """Raises GraphDispatchError for node 'a', succeeds for everything else."""

    async def dispatch(
        self, node: Any, request: Any, ctx: Any, first_token_cb: Any = None
    ) -> str:
        if request.node_id == "a":
            raise GraphDispatchError("simulated dispatch failure")
        return ""


@pytest.mark.asyncio
async def test_success_placeholder_is_list_for_messages_channel() -> None:
    parsed = _messages_chain_parsed(with_pool=False)
    executor = TraceExecutor(parsed, credit_issuer=_EchoIssuer())
    async with asyncio.TaskGroup():
        result = await executor.run(parsed.traces[0])
    assert result.trace_id == "t1"


@pytest.mark.asyncio
async def test_failure_sentinel_is_list_for_messages_channel() -> None:
    parsed = _messages_chain_parsed(with_pool=True)
    executor = TraceExecutor(parsed, credit_issuer=_FailFirstIssuer())
    async with asyncio.TaskGroup():
        result = await executor.run(parsed.traces[0])
    assert result.trace_id == "t1"


@pytest.mark.asyncio
async def test_failure_without_pool_keeps_orphan_semantics() -> None:
    """Non-segment-store parses keep the historical orphan-then-error behavior."""
    parsed = _messages_chain_parsed(with_pool=False)
    executor = TraceExecutor(parsed, credit_issuer=_FailFirstIssuer())
    with pytest.raises(ExceptionGroup) as excinfo:
        async with asyncio.TaskGroup():
            await executor.run(parsed.traces[0])
    from aiperf.graph.channel_store import ChannelOrphanedError

    # The dispatch failure is contained ("continuing past the failed turn");
    # what surfaces is the ORPHANED downstream channel -- the exact historical
    # orphan-then-error semantics this test locks.
    assert excinfo.group_contains(ChannelOrphanedError)


class _RefuseFirstIssuer(_EchoIssuer):
    """Refuses node 'a' the way the issuer stop gate does at duration end."""

    async def dispatch(
        self, node: Any, request: Any, ctx: Any, first_token_cb: Any = None
    ) -> str:
        if request.node_id == "a":
            from aiperf.graph.credit_dispatch_adapter import CreditIssueRefusedError

            raise CreditIssueRefusedError("refused by issuer stop gate")
        return ""


@pytest.mark.asyncio
async def test_issuer_refusal_stops_trace_even_with_pool() -> None:
    """Refusal is a trace-stop, never sentinel-continued.

    Contrast with test_failure_sentinel_is_list_for_messages_channel: a plain
    dispatch failure on the same parse continues with omission; a refusal must
    unwind so duration-end traces stop instead of sentinel-spinning through
    their remaining nodes.
    """
    parsed = _messages_chain_parsed(with_pool=True)
    executor = TraceExecutor(parsed, credit_issuer=_RefuseFirstIssuer())
    with pytest.raises(ExceptionGroup) as excinfo:
        async with asyncio.TaskGroup():
            await executor.run(parsed.traces[0])
    from aiperf.graph.credit_dispatch_adapter import CreditIssueRefusedError

    # The unwind must carry the REFUSAL, proving the stop was the issuer gate
    # (not a sentinel-continued dispatch failure or unrelated error).
    assert excinfo.group_contains(CreditIssueRefusedError)
