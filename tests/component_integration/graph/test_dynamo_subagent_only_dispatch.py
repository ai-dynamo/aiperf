# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A parent + subagent dynamo trace dispatches every flat node exactly once through the schedule-plane ``TraceExecutor``."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aiperf.dataset.graph.models import LlmNode
from aiperf.dataset.graph.workload_detect import parse_graph_workload
from aiperf.graph.executor import TraceExecutor
from tests.component_integration.graph.conftest import (
    dynamo_request_end,
    dynamo_run,
    write_dynamo_jsonl,
)

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]


@pytest.fixture
def subagent_fixture(tmp_path: Path) -> Path:
    """Parent A (3 turns) plus child B whose turns interleave A's K=2..3 window."""
    # B's turns (ts 1150 / 1170) fall between A's K=2 (1100) and K=3 (1200)
    # recorded points, so the flat lowering chains them by finished-before order.
    return write_dynamo_jsonl(
        tmp_path / "subagent_flat.jsonl",
        [
            dynamo_request_end(ts=1000, session_id="A", hashes=[11, 22]),
            dynamo_request_end(ts=1100, session_id="A", hashes=[11, 22, 33]),
            dynamo_request_end(
                ts=1150, session_id="B", parent_session_id="A", hashes=[90, 91]
            ),
            dynamo_request_end(
                ts=1170, session_id="B", parent_session_id="A", hashes=[90, 91, 92]
            ),
            dynamo_request_end(ts=1200, session_id="A", hashes=[11, 22, 33, 44]),
        ],
    )


class _ScalarIssuer:
    """Stub credit issuer returning the graph dispatch result contract."""

    # That placeholder is `""`, the exact value that would unwind a list-reducer
    # channel if a placeholder write ever reached one.

    def __init__(self) -> None:
        self.n = 0

    async def dispatch(
        self, node, request, ctx, **kw
    ) -> tuple[str, int | None, float | None, float | None]:
        self.n += 1
        return "", None, None, None


async def test_flat_parent_child_dispatches_through_executor(
    subagent_fixture: Path,
) -> None:
    """The flat parent+child graph runs to completion through the real executor with all five nodes dispatched."""
    parsed = parse_graph_workload(dynamo_run(subagent_fixture), subagent_fixture)

    # Coverage guard: the flat lowering really produced parent + child session
    # LlmNodes in ONE graph, each writing a declared per-node scratch channel.
    llm_nodes = {
        nid: n for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)
    }
    assert len(llm_nodes) == 5, sorted(parsed.graph.nodes)
    assert "messages" not in parsed.graph.state
    for nid, node in llm_nodes.items():
        assert node.output == f"{nid}_out"
        assert f"{nid}_out" in parsed.graph.state

    # Drive the REAL executor with a scalar-returning issuer over every trace.
    # Must complete without raising: every scalar placeholder write lands on a
    # scalar-tolerant scratch channel.
    issuer = _ScalarIssuer()
    ex = TraceExecutor(parsed, credit_issuer=issuer)
    async with asyncio.TaskGroup():
        for trace in parsed.traces:
            await ex.run(trace)

    assert issuer.n == 5, (
        f"expected 3 parent + 2 child dispatches; got {issuer.n} -- some flat "
        "node never fired through the executor"
    )
