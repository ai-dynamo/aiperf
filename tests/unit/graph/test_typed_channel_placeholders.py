# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Type-correct channel placeholders on the LlmNode credit path."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pytest import param

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
from aiperf.dataset.graph.segment_trie.pool import SegmentPool
from aiperf.graph.channel_store import ChannelOrphanedError
from aiperf.graph.credit_dispatch_adapter import (
    CreditIssueRefusedError,
    GraphDispatchError,
)
from aiperf.graph.executor import TraceExecutor


def _messages_chain_parsed(*, with_pool: bool) -> ParsedGraph:
    """a -> b where ``a_out`` is a MESSAGES channel reduced by ``add_messages``."""
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
    ) -> tuple[str, int | None, float | None, float | None]:
        return "", None, None, None


class _FailFirstIssuer(_EchoIssuer):
    """Raises GraphDispatchError for node 'a', succeeds for everything else."""

    async def dispatch(
        self, node: Any, request: Any, ctx: Any, first_token_cb: Any = None
    ) -> tuple[str, int | None, float | None, float | None]:
        if request.node_id == "a":
            raise GraphDispatchError("simulated dispatch failure")
        return "", None, None, None


class _RefuseFirstIssuer(_EchoIssuer):
    """Refuses node 'a' the way the issuer stop gate does at duration end."""

    async def dispatch(
        self, node: Any, request: Any, ctx: Any, first_token_cb: Any = None
    ) -> tuple[str, int | None, float | None, float | None]:
        if request.node_id == "a":
            raise CreditIssueRefusedError("refused by issuer stop gate")
        return "", None, None, None


@pytest.mark.parametrize(
    ("with_pool", "issuer_cls"),
    [
        param(False, _EchoIssuer, id="success-placeholder-no-pool"),
        param(True, _FailFirstIssuer, id="failure-sentinel-with-pool"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_messages_channel_writes_are_list_typed_and_trace_completes(
    with_pool: bool, issuer_cls: type[_EchoIssuer]
) -> None:
    """The run completes because both placeholder and failure sentinel are ``[]``, not a str."""
    # ``add_messages`` rejects a non-list write, so a str placeholder/sentinel would
    # blow up the reducer instead of letting the trace finish.
    parsed = _messages_chain_parsed(with_pool=with_pool)
    executor = TraceExecutor(parsed, credit_issuer=issuer_cls())

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

    # The dispatch failure is contained ("continuing past the failed turn");
    # what surfaces is the ORPHANED downstream channel -- the exact historical
    # orphan-then-error semantics this test locks.
    assert excinfo.group_contains(ChannelOrphanedError)


@pytest.mark.asyncio
async def test_issuer_refusal_stops_trace_even_with_pool() -> None:
    """Refusal is a trace-stop, never sentinel-continued."""
    parsed = _messages_chain_parsed(with_pool=True)
    executor = TraceExecutor(parsed, credit_issuer=_RefuseFirstIssuer())

    with pytest.raises(ExceptionGroup) as excinfo:
        async with asyncio.TaskGroup():
            await executor.run(parsed.traces[0])

    # The unwind must carry the REFUSAL, proving the stop was the issuer gate
    # (not a sentinel-continued dispatch failure or unrelated error).
    assert excinfo.group_contains(CreditIssueRefusedError)
