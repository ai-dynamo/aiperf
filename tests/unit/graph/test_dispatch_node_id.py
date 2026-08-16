# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dispatch body reports the id the SCHEDULER fired, not an identity scan."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aiperf.dataset.graph.models import (
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
from aiperf.graph.executor import TraceExecutor


class _RecordingIssuer:
    """Records every dispatched ``request.node_id``."""

    def __init__(self) -> None:
        self.node_ids: list[str] = []

    async def dispatch(
        self, node: Any, request: Any, ctx: Any, first_token_cb: Any = None
    ) -> tuple[str, int | None, float | None, float | None]:
        self.node_ids.append(request.node_id)
        return "", None, None, None


@pytest.mark.asyncio
async def test_shared_node_struct_dispatches_under_each_distinct_id() -> None:
    """Two ids bound to the SAME frozen struct must each report their own id."""
    # msgspec structs are frozen and freely interned by adapters, so one
    # instance legitimately appears under several node ids. Recovering the id
    # by identity-scanning ``graph.nodes`` would report the FIRST match for
    # both, silently collapsing two turns' spans onto one node.
    shared = LlmNode(prompt=[{"role": "user", "content": "q"}], output="out")
    graph = GraphRecord(
        state={
            "out": ChannelSpec(
                type=ChannelType.MESSAGES, reducer=ReducerName.ADD_MESSAGES
            )
        },
        nodes={"x": shared, "y": shared},
        edges=[
            StaticEdge(source="START", target="x"),
            StaticEdge(source="START", target="y"),
            StaticEdge(source="x", target="END"),
            StaticEdge(source="y", target="END"),
        ],
    )
    parsed = ParsedGraph(
        graph=graph, traces=[TraceRecord(id="t1")], segment_pool=SegmentPool()
    )
    issuer = _RecordingIssuer()
    executor = TraceExecutor(parsed, credit_issuer=issuer)

    async with asyncio.TaskGroup():
        await executor.run(parsed.traces[0])

    assert sorted(issuer.node_ids) == ["x", "y"]
