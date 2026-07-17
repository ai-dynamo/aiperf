# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A parent + subagent dynamo trace dispatches through the schedule-plane
``TraceExecutor``.

In the flat trie IR every session's turns are ordinary ``LlmNode``s writing
their scalar
dispatch placeholder to a per-node ``{node_id}_out`` scratch channel (TEXT /
overwrite, scalar-tolerant), with concurrency emergent from the recorded
intervals. This drives the REAL schedule-plane parse (``parse_graph_workload``
-> ``from_dynamo_trace`` trie lowering) through a ``TraceExecutor`` with a
scalar-returning stub issuer (the exact ``""`` the real
``CreditDispatchAdapter`` returns) and asserts the executor COMPLETES with
every node -- parent AND child session -- dispatched exactly once.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph.models import LlmNode
from aiperf.dataset.graph.workload_detect import parse_graph_workload
from aiperf.graph.executor import TraceExecutor
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import make_run_from_cli

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]

_SEED = 1234


def _request_end(
    *,
    ts: int,
    session_id: str,
    hashes: list[int],
    parent_session_id: str | None = None,
) -> dict[str, Any]:
    """One ``dynamo.request.trace.v1`` ``request_end`` with recorded replay hashes.

    One 16-token block per replay hash so ``input_length == 16 * len(hashes)``
    stays block-aligned for the parse-time alignment gate.
    """
    input_length = 16 * len(hashes)
    ctx: dict[str, Any] = {"session_id": session_id}
    if parent_session_id is not None:
        ctx["parent_session_id"] = parent_session_id
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": ctx,
        "request": {
            "request_id": f"{session_id}-{ts}",
            "model": "m",
            "input_tokens": input_length,
            "output_tokens": 16,
            "cached_tokens": 0,
            "replay": {
                "trace_block_size": 16,
                "input_length": input_length,
                "input_sequence_hashes": hashes,
            },
        },
    }


@pytest.fixture
def subagent_fixture(tmp_path: Path) -> Path:
    """Parent A (3 turns) + child B whose turns interleave A's K=2..3 window.

    B's turns (ts 1150 / 1170) fall between A's K=2 (1100) and K=3 (1200)
    recorded points, so the flat lowering chains them by finished-before order,
    with every scalar placeholder write landing on a per-node scratch channel.
    """
    p = tmp_path / "subagent_flat.jsonl"
    records = [
        _request_end(ts=1000, session_id="A", hashes=[11, 22]),
        _request_end(ts=1100, session_id="A", hashes=[11, 22, 33]),
        _request_end(ts=1150, session_id="B", parent_session_id="A", hashes=[90, 91]),
        _request_end(
            ts=1170, session_id="B", parent_session_id="A", hashes=[90, 91, 92]
        ),
        _request_end(ts=1200, session_id="A", hashes=[11, 22, 33, 44]),
    ]
    with p.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")
    return p


def _parse(fixture: Path):
    run = make_run_from_cli(
        CLIConfig(
            model_names=["m"],
            endpoint_type=EndpointType.CHAT,
            streaming=False,
            url="http://localhost:8000",
            input_file=str(fixture),
            random_seed=_SEED,
        )
    )
    return parse_graph_workload(run, fixture)


class _ScalarIssuer:
    """Stub credit issuer returning the scalar placeholder the real
    ``CreditDispatchAdapter.dispatch`` returns (``""``) -- the exact value that
    would unwind a list-reducer channel if a placeholder write ever reached one."""

    def __init__(self) -> None:
        self.n = 0

    async def dispatch(self, node, request, ctx, **kw) -> str:
        self.n += 1
        return ""


async def test_flat_parent_child_dispatches_through_executor(
    subagent_fixture: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flat parent+child graph runs to completion through the real executor."""

    parsed = _parse(subagent_fixture)

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
