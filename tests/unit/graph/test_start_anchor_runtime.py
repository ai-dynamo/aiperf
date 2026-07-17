# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime proof: start-anchored edges dispatch children at the predecessor's
DISPATCH instant (firing-gate clear), not its completion.

Drives the REAL ``TraceExecutor`` on a ``VirtualClock`` over weka trie graphs
whose overlap geometry produces ``StaticEdge.delay_after_predecessor_start_us``
edges (Task 3 stamping). The harness (``_STUB_CALLBACKS``, ``_VTimeIssuer``,
``_drive_virtual``) mirrors ``test_executor_runs_weka.py``.

Contract:
1. Recorded-speed replay dispatches at exactly the recorded start instants.
2. A slowed parent keeps DISPATCH-anchored children fixed and moves the
   end-anchored child to the parent's (later) finish, with NO cycle RuntimeError
   even though a dispatch-anchored child finished long before the parent.
3. A delayed parent DISPATCH shifts a dispatch-anchored child by exactly the
   same amount (dispatch-relative delay preserved).
4. ``AIPERF_GRAPH_IGNORE_EDGE_DELAYS`` short-circuits the gate uniformly while
   dispatch-time scheduling still fires every child exactly once.
"""

import asyncio
from typing import Any

import msgspec
import pytest

from aiperf.common.clock import VirtualClock
from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
from aiperf.dataset.graph.adapters.weka.trie_build import (
    ReconCallbacks,
    build_trie_graph,
)
from aiperf.dataset.graph.models import TraceRecord
from aiperf.graph.executor import TraceExecutor

_BLOCK_SIZE = 64


def _stub_decode_block_tokens(hash_ids: list[int]) -> list[int]:
    out: list[int] = []
    for h in hash_ids:
        out.extend(range(h * 100, h * 100 + _BLOCK_SIZE))
    return out


def _stub_partial_tail_tokens(n_tokens: int, seed: str) -> list[int]:
    base = sum(ord(c) for c in seed) * 1000
    return list(range(base, base + n_tokens))


def _stub_decode_tokens_to_text(tokens: list[int]) -> str:
    return "|".join(str(t) for t in tokens)


_STUB_CALLBACKS = ReconCallbacks(
    decode_block_tokens=_stub_decode_block_tokens,
    sample_partial_tail_tokens=_stub_partial_tail_tokens,
    decode_tokens_to_text=_stub_decode_tokens_to_text,
)


class _VTimeIssuer:
    """Records each node's VIRTUAL dispatch time, then consumes the node's
    recorded ``api_time`` in virtual time.

    ``api_by_id`` maps ``request.node_id`` -> recorded processing seconds. The
    executor records the node's finish as ``dispatch_start + api_time`` (the
    issuer's virtual sleep), so an end-anchored successor's firing gate clears at
    the recorded end-to-start instant while a start-anchored successor clears at
    ``dispatch + delay``.
    """

    def __init__(self, clock: Any, api_by_id: dict[str, float]) -> None:
        self._clock = clock
        self._api = api_by_id
        self.dispatched_at: dict[str, float] = {}
        self.dispatched: list[str] = []

    async def dispatch(
        self,
        node: Any,
        request: Any,
        ctx: Any,
        **kwargs: Any,
    ) -> str:
        nid = request.node_id
        self.dispatched.append(nid)
        self.dispatched_at[nid] = self._clock.now_ns() / 1e9
        api_s = self._api.get(nid, 0.0)
        if api_s > 0.0:
            await self._clock.sleep_ns(int(api_s * 1e9))
        return f"placeholder::{nid}"


async def _drive_virtual(clock: Any, task: Any) -> Any:
    """Pump a virtual-clock replay: drain ready callbacks, then fast-forward sim
    time to the earliest parked waiter whenever the loop goes idle."""
    loop = asyncio.get_running_loop()
    ready = loop._ready  # noqa: SLF001 -- idle detection for the pump
    while not task.done():
        while ready and not task.done():
            await asyncio.sleep(0)
        if task.done():
            break
        nxt = clock.peek_min_waiter_ns()
        if nxt is None:
            await asyncio.sleep(0)
            if not ready and clock.peek_min_waiter_ns() is None and not task.done():
                raise RuntimeError("virtual-time replay stalled")
            continue
        await clock.advance_to(nxt)
    return task.result()


# --- fixtures -------------------------------------------------------------

# P: t=0 api=8.0 (long, spawner); C: subagent first at t=2.5 (P in flight);
# Q: chain-overlap at t=5.0 (P in flight); R: t=9.0 (after P ends, end-anchored)
_OVERLAP_TRACE = {
    "id": "start_anchor", "models": ["M"], "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "n", "model": "M", "in": 128, "out": 64,
         "hash_ids": [1, 2], "api_time": 8.0, "stop": "tool_use"},
        {"t": 2.0, "type": "subagent", "agent_id": "a1",
         "subagent_type": "Explore", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 2.5, "type": "n", "model": "M", "in": 128, "out": 16,
              "hash_ids": [50, 51], "api_time": 1.0},
         ]},
        {"t": 5.0, "type": "n", "model": "M", "in": 192, "out": 32,
         "hash_ids": [1, 2, 3], "api_time": 1.0},
        {"t": 9.0, "type": "n", "model": "M", "in": 256, "out": 32,
         "hash_ids": [1, 2, 3, 4], "api_time": 0.5},
    ],
}  # fmt: skip

# Node ids the trie build assigns to _OVERLAP_TRACE (verified against the graph).
_P = "start_anchor:0"
_C = "a1:0"
_Q = "start_anchor:1"
_R = "start_anchor:2"

# P1 (t=0, api=1.0), P2 (t=2.0, api=8.0, tool_use), subagent at t=3.0 with C
# (t=4.0, api=1.0): C anchors to P2's DISPATCH with D=2.0; P2 anchors to P1's
# finish with recorded end-to-start delay 1.0.
_CHAIN_TRACE = {
    "id": "start_anchor_chain", "models": ["M"], "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "n", "model": "M", "in": 128, "out": 64,
         "hash_ids": [1, 2], "api_time": 1.0},
        {"t": 2.0, "type": "n", "model": "M", "in": 192, "out": 32,
         "hash_ids": [1, 2, 3], "api_time": 8.0, "stop": "tool_use"},
        {"t": 3.0, "type": "subagent", "agent_id": "a1",
         "subagent_type": "Explore", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 4.0, "type": "n", "model": "M", "in": 128, "out": 16,
              "hash_ids": [50, 51], "api_time": 1.0},
         ]},
    ],
}  # fmt: skip

# Node ids the trie build assigns to _CHAIN_TRACE (verified against the graph).
_P1 = "start_anchor_chain:0"
_P2 = "start_anchor_chain:1"
_CHAIN_C = "a1:0"


def _build(raw: dict) -> Any:
    """Build a trie graph from a raw weka trace dict and wire a bare trace."""
    trace = WekaTrace.model_validate(raw)
    parsed, pool = build_trie_graph(trace, callbacks=_STUB_CALLBACKS)
    bare = TraceRecord(id=trace.id)
    return msgspec.structs.replace(parsed, traces=[bare], segment_pool=pool)


async def _run_virtual(parsed: Any, api_by_id: dict[str, float]) -> _VTimeIssuer:
    """Drive ``parsed``'s traces on a VirtualClock and return the issuer."""
    clock = VirtualClock()
    issuer = _VTimeIssuer(clock, api_by_id)
    executor = TraceExecutor(parsed, credit_issuer=issuer, clock=clock)

    async def _phase() -> None:
        async with asyncio.TaskGroup():
            for trace in parsed.traces:
                await executor.run(trace)

    phase_task = asyncio.ensure_future(_phase())
    await _drive_virtual(clock, phase_task)
    return issuer


# --- tests ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_recorded_speed_virtual_replay_matches_recorded_starts():
    """At recorded speed the four nodes dispatch at exactly 0.0/2.5/5.0/9.0s.

    C and Q are start-anchored to P's dispatch (0.0) at +2.5 / +5.0. R is
    end-anchored: it waits for P's finish (0.0 + api 8.0) + recorded delay 1.0.
    """
    parsed = _build(_OVERLAP_TRACE)
    api_by_id = {_P: 8.0, _C: 1.0, _Q: 1.0, _R: 0.5}

    issuer = await _run_virtual(parsed, api_by_id)

    assert set(issuer.dispatched_at) == {_P, _C, _Q, _R}
    assert issuer.dispatched_at[_P] == pytest.approx(0.0, abs=1e-3)
    assert issuer.dispatched_at[_C] == pytest.approx(2.5, abs=1e-3)
    assert issuer.dispatched_at[_Q] == pytest.approx(5.0, abs=1e-3)
    assert issuer.dispatched_at[_R] == pytest.approx(9.0, abs=1e-3)


@pytest.mark.asyncio
async def test_slowed_parent_moves_children_with_dispatch_not_wall_clock():
    """Doubling P's server time keeps DISPATCH-anchored children fixed.

    P inflated to api=16.0. C and Q are anchored to P's DISPATCH (0.0), so they
    stay at 2.5 / 5.0. R is end-anchored: it fires at P's finish (16.0) +
    recorded end-to-start delay (1.0) = 17.0.

    Crucially C finishes at 3.5 -- long before P finishes at 16.0. Start-anchored
    successors are excluded from ``successors_after``, so P's completion does NOT
    re-schedule C into the cycle guard. Reaching the asserts proves no
    RuntimeError was raised.
    """
    parsed = _build(_OVERLAP_TRACE)
    api_by_id = {_P: 16.0, _C: 1.0, _Q: 1.0, _R: 0.5}

    issuer = await _run_virtual(parsed, api_by_id)

    assert issuer.dispatched_at[_C] == pytest.approx(2.5, abs=1e-3)
    assert issuer.dispatched_at[_Q] == pytest.approx(5.0, abs=1e-3)
    assert issuer.dispatched_at[_R] == pytest.approx(17.0, abs=1e-3)


@pytest.mark.asyncio
async def test_delayed_parent_dispatch_shifts_children():
    """Slowing P1 pushes P2's DISPATCH later; the start-anchored child moves with
    it by exactly the same amount (dispatch-relative delay preserved).

    Recorded: P1=0->finish 1.0; P2 dispatch = 1.0 + delay 1.0 = 2.0; C =
    P2 dispatch 2.0 + D 2.0 = 4.0. Inflating P1 api to 5.0 moves P2's finish to
    5.0, so P2 dispatches at 5.0 + 1.0 = 6.0 and C at 6.0 + 2.0 = 8.0 -- C shifts
    by the same +4.0 as P2.
    """
    parsed = _build(_CHAIN_TRACE)
    api_by_id = {_P1: 5.0, _P2: 8.0, _CHAIN_C: 1.0}

    issuer = await _run_virtual(parsed, api_by_id)

    assert set(issuer.dispatched_at) == {_P1, _P2, _CHAIN_C}
    assert issuer.dispatched_at[_P1] == pytest.approx(0.0, abs=1e-3)
    assert issuer.dispatched_at[_P2] == pytest.approx(6.0, abs=1e-3)
    assert issuer.dispatched_at[_CHAIN_C] == pytest.approx(8.0, abs=1e-3)


@pytest.mark.asyncio
async def test_ignore_edge_delays_fires_children_at_parent_dispatch(monkeypatch):
    """``AIPERF_GRAPH_IGNORE_EDGE_DELAYS`` collapses the gate but keeps
    dispatch-time scheduling.

    With delays ignored, C and Q dispatch at the same virtual instant P
    dispatches (0.0) -- scheduling still happens at P's dispatch, so P precedes
    both in dispatch order -- and all four nodes dispatch exactly once.
    """
    monkeypatch.setattr(Environment.GRAPH, "IGNORE_EDGE_DELAYS", True, raising=False)

    parsed = _build(_OVERLAP_TRACE)
    api_by_id = {_P: 8.0, _C: 1.0, _Q: 1.0, _R: 0.5}

    issuer = await _run_virtual(parsed, api_by_id)

    from collections import Counter

    counts = Counter(issuer.dispatched)
    assert set(counts) == {_P, _C, _Q, _R}
    assert all(n == 1 for n in counts.values()), (
        f"every node must dispatch exactly once; got {counts}"
    )
    assert issuer.dispatched_at[_P] == pytest.approx(0.0, abs=1e-3)
    assert issuer.dispatched_at[_C] == pytest.approx(0.0, abs=1e-3)
    assert issuer.dispatched_at[_Q] == pytest.approx(0.0, abs=1e-3)
    # Scheduling still happens at P's dispatch, so P precedes both children.
    order = issuer.dispatched
    assert order.index(_P) < order.index(_C)
    assert order.index(_P) < order.index(_Q)
