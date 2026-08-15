# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Containment of ``GraphDispatchError`` must never itself raise.

The sentinel write in ``TraceExecutor._handle_node_exception`` targets the
failed node's own ``write_channels``. When such a channel uses the OVERWRITE
reducer and another node already wrote it, the store's genuine duplicate-write
guard would escalate a CONTAINED per-node failure into a trace-fatal error.
"""

from __future__ import annotations

import pytest

from aiperf.dataset.graph.models import (
    ChannelSpec,
    ChannelType,
    GraphRecord,
    LlmNode,
    ParsedGraph,
    ReducerName,
    TraceRecord,
)
from aiperf.dataset.graph.segment_trie.pool import SegmentPool
from aiperf.graph.channel_store import VersionedChannelStore
from aiperf.graph.context import _TraceContext
from aiperf.graph.credit_dispatch_adapter import GraphDispatchError
from aiperf.graph.executor import TraceExecutor
from aiperf.graph.reducers import OverwriteConflictError


def _build(
    channel: str, *, channel_type: ChannelType
) -> tuple[TraceExecutor, _TraceContext, LlmNode]:
    """Two nodes writing one shared channel, with a segment pool attached."""
    node_a = LlmNode(prompt=["a"], output=channel)
    node_b = LlmNode(prompt=["b"], output=channel)
    graph = GraphRecord(
        nodes={"a": node_a, "b": node_b},
        edges=[],
        state={
            channel: ChannelSpec(type=channel_type, reducer=ReducerName.OVERWRITE),
        },
    )
    parsed = ParsedGraph(
        graph=graph,
        traces=[TraceRecord(id="t")],
        segment_pool=SegmentPool(),
    )
    executor = TraceExecutor(parsed)
    store = VersionedChannelStore(
        initial={},
        channel_specs=dict(graph.state),
        producers_per_channel={channel: 2},
    )
    ctx = _TraceContext(trace=parsed.traces[0], store=store)
    return executor, ctx, node_b


@pytest.mark.parametrize(
    "channel_type",
    [ChannelType.TEXT, ChannelType.MESSAGES],
)  # fmt: skip
def test_handle_node_exception_overwrite_channel_already_written_contains(
    channel_type: ChannelType,
) -> None:
    """Containment tolerates an already-written OVERWRITE channel."""
    executor, ctx, node_b = _build("shared", channel_type=channel_type)
    ctx.store.write(["shared"], "from-a", writer_node_id="a")

    result = executor._handle_node_exception(
        "b",
        node_b,
        ctx=ctx,
        exc=GraphDispatchError("worker died"),
    )

    assert result is not None
    # The prior writer's value must survive: the sentinel must not clobber it.
    assert ctx.store.snapshot()["shared"] == "from-a"


def test_write_overwrite_channel_twice_still_raises() -> None:
    """The normal (non-containment) duplicate-write invariant is unchanged."""
    _, ctx, _ = _build("shared", channel_type=ChannelType.TEXT)
    ctx.store.write(["shared"], "from-a", writer_node_id="a")
    with pytest.raises(OverwriteConflictError):
        ctx.store.write(["shared"], "from-b", writer_node_id="b")
