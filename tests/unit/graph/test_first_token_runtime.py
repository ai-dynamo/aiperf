# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime proof: post-TTFT first-token-anchored edges gate a successor on the
predecessor's OBSERVED first token, falling back to the dispatch anchor when the
predecessor terminates without streaming one.

Drives the REAL ``TraceExecutor`` on a ``VirtualClock`` over a weka trie graph
whose overlap geometry produces a ``StaticEdge`` carrying BOTH
``delay_after_predecessor_start_us`` (D) and
``delay_after_predecessor_first_token_us`` (D') (Task 2 lowering). The harness
(``_STUB_CALLBACKS``, ``_drive_virtual``) mirrors ``test_start_anchor_runtime.py``;
``_TTFTIssuer`` extends ``_VTimeIssuer`` so a node with a recorded ttft sleeps
``ttft`` virtual seconds, invokes the ``first_token_cb`` it received as a dispatch
kwarg, then sleeps the remainder.

Contract:
1. Recorded-speed replay reproduces the recorded starts (observed == fallback).
2. Inflating the parent's ttft moves the first-token-anchored child by exactly
   the inflation while the pre-TTFT dispatch-anchored child stays put.
3. A parent that errors before its first token gates the child at the dispatch
   fallback (``_finalize_node`` latches the event; no wall entry).
4. ``AIPERF_GRAPH_IGNORE_EDGE_DELAYS`` short-circuits the wait entirely.
5. A parent that resolves WITHOUT streaming a first token falls back; a late /
   duplicate stamp is a no-op (wall not overwritten).
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
from aiperf.graph.credit_dispatch_adapter import GraphDispatchError
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
    recorded ``api_time`` in virtual time (mirrors ``test_start_anchor_runtime``).
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
        first_token_cb: Any = None,
        **kwargs: Any,
    ) -> str:
        nid = request.node_id
        self.dispatched.append(nid)
        self.dispatched_at[nid] = self._clock.now_ns() / 1e9
        api_s = self._api.get(nid, 0.0)
        if api_s > 0.0:
            await self._clock.sleep_ns(int(api_s * 1e9))
        return f"placeholder::{nid}"


class _TTFTIssuer(_VTimeIssuer):
    """A node with a recorded ttft streams a first token at ``ttft`` virtual
    seconds (invoking the received ``first_token_cb``), then finishes the
    remaining ``api - ttft`` virtual seconds. Nodes without a recorded ttft
    behave exactly like ``_VTimeIssuer`` (sleep ``api``, never stamp)."""

    def __init__(
        self, clock: Any, api_by_id: dict[str, float], ttft_by_id: dict[str, float]
    ) -> None:
        super().__init__(clock, api_by_id)
        self._ttft = ttft_by_id

    async def dispatch(
        self,
        node: Any,
        request: Any,
        ctx: Any,
        first_token_cb: Any = None,
        **kwargs: Any,
    ) -> str:
        nid = request.node_id
        self.dispatched.append(nid)
        self.dispatched_at[nid] = self._clock.now_ns() / 1e9
        ttft = self._ttft.get(nid)
        api = self._api.get(nid, 0.0)
        if ttft is not None:
            await self._clock.sleep_ns(int(ttft * 1e9))
            if first_token_cb is not None:
                first_token_cb()
            await self._clock.sleep_ns(int(max(0.0, api - ttft) * 1e9))
        elif api > 0.0:
            await self._clock.sleep_ns(int(api * 1e9))
        return f"placeholder::{nid}"


class _ErrorBeforeFirstTokenIssuer(_TTFTIssuer):
    """The named node raises ``GraphDispatchError`` after ``error_after`` virtual
    seconds WITHOUT streaming a first token (the trie failure-sentinel path); all
    other nodes behave like ``_TTFTIssuer``."""

    def __init__(
        self,
        clock: Any,
        api_by_id: dict[str, float],
        ttft_by_id: dict[str, float],
        error_node: str,
        error_after: float,
    ) -> None:
        super().__init__(clock, api_by_id, ttft_by_id)
        self._error_node = error_node
        self._error_after = error_after

    async def dispatch(
        self,
        node: Any,
        request: Any,
        ctx: Any,
        first_token_cb: Any = None,
        **kwargs: Any,
    ) -> str:
        nid = request.node_id
        if nid == self._error_node:
            self.dispatched.append(nid)
            self.dispatched_at[nid] = self._clock.now_ns() / 1e9
            await self._clock.sleep_ns(int(self._error_after * 1e9))
            raise GraphDispatchError(f"simulated dispatch failure for {nid!r}")
        return await super().dispatch(node, request, ctx, first_token_cb, **kwargs)


class _DoubleStampTTFTIssuer(_TTFTIssuer):
    """Invokes ``first_token_cb`` TWICE at the first-token instant for the named
    node -- the second call must be a graceful no-op (guard in the stamp)."""

    def __init__(
        self,
        clock: Any,
        api_by_id: dict[str, float],
        ttft_by_id: dict[str, float],
        double_node: str,
    ) -> None:
        super().__init__(clock, api_by_id, ttft_by_id)
        self._double_node = double_node

    async def dispatch(
        self,
        node: Any,
        request: Any,
        ctx: Any,
        first_token_cb: Any = None,
        **kwargs: Any,
    ) -> str:
        nid = request.node_id
        ttft = self._ttft.get(nid)
        if nid == self._double_node and ttft is not None:
            self.dispatched.append(nid)
            self.dispatched_at[nid] = self._clock.now_ns() / 1e9
            api = self._api.get(nid, 0.0)
            await self._clock.sleep_ns(int(ttft * 1e9))
            if first_token_cb is not None:
                first_token_cb()
                first_token_cb()  # late duplicate: must be a no-op
            await self._clock.sleep_ns(int(max(0.0, api - ttft) * 1e9))
            return f"placeholder::{nid}"
        return await super().dispatch(node, request, ctx, first_token_cb, **kwargs)


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


# --- fixture --------------------------------------------------------------

# P (ttft_anchor:0): streaming, t=0, ttft=2.0, api=8.0 (spawner).
# C_pre (a1:0): subagent inner request at t=1.0 -- PRE-TTFT (1.0 < 2.0), so a
#   PURE dispatch anchor (start=1.0e6, ft=None).
# C_post (a2:0): subagent inner request at t=4.0 -- POST-TTFT (4.0 >= 2.0), so a
#   first-token-refined anchor (start=4.0e6, ft=D-ttft=2.0e6).
# tail (ttft_anchor:1): end-anchored at t=9.0.
_TTFT_TRACE = {
    "id": "ttft_anchor", "models": ["M"], "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "s", "ttft": 2.0, "api_time": 8.0, "in": 128, "out": 64,
         "hash_ids": [1, 2], "stop": "tool_use", "model": "M"},
        {"t": 0.5, "type": "subagent", "agent_id": "a1",
         "subagent_type": "Explore", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 1.0, "type": "n", "model": "M", "in": 128, "out": 16,
              "hash_ids": [50, 51], "api_time": 1.0},
         ]},
        {"t": 3.5, "type": "subagent", "agent_id": "a2",
         "subagent_type": "Explore", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 4.0, "type": "n", "model": "M", "in": 128, "out": 16,
              "hash_ids": [60, 61], "api_time": 1.0},
         ]},
        {"t": 9.0, "type": "n", "model": "M", "in": 256, "out": 32,
         "hash_ids": [1, 2, 3, 4], "api_time": 0.5},
    ],
}  # fmt: skip

# Node ids the trie build assigns to _TTFT_TRACE (verified against the graph):
# top-level leaves scope to the trace id (skipping subagent markers), subagent
# inner leaves scope to their recorded agent_id.
_P = "ttft_anchor:0"
_C_PRE = "a1:0"
_C_POST = "a2:0"
_TAIL = "ttft_anchor:1"


def _build(raw: dict) -> Any:
    """Build a trie graph from a raw weka trace dict and wire a bare trace."""
    trace = WekaTrace.model_validate(raw)
    parsed, pool = build_trie_graph(trace, callbacks=_STUB_CALLBACKS)
    bare = TraceRecord(id=trace.id)
    return msgspec.structs.replace(parsed, traces=[bare], segment_pool=pool)


async def _run_virtual(parsed: Any, issuer: Any) -> Any:
    """Drive ``parsed``'s traces on ``issuer``'s VirtualClock and return it."""
    clock = issuer._clock
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
async def test_recorded_speed_reproduces_recorded_starts():
    """At recorded speed observed==fallback: C_post fires at 4.0 (first token at
    2.0 + D'=2.0), C_pre at 1.0 (pure dispatch anchor), tail at 9.0."""
    parsed = _build(_TTFT_TRACE)
    clock = VirtualClock()
    issuer = _TTFTIssuer(
        clock,
        api_by_id={_P: 8.0, _C_PRE: 1.0, _C_POST: 1.0, _TAIL: 0.5},
        ttft_by_id={_P: 2.0},
    )

    await _run_virtual(parsed, issuer)

    assert set(issuer.dispatched_at) == {_P, _C_PRE, _C_POST, _TAIL}
    assert issuer.dispatched_at[_P] == pytest.approx(0.0, abs=1e-3)
    assert issuer.dispatched_at[_C_PRE] == pytest.approx(1.0, abs=1e-3)
    assert issuer.dispatched_at[_C_POST] == pytest.approx(4.0, abs=1e-3)
    assert issuer.dispatched_at[_TAIL] == pytest.approx(9.0, abs=1e-3)


@pytest.mark.asyncio
async def test_inflated_ttft_moves_first_token_child_by_the_inflation():
    """Inflating P's ttft 2.0 -> 5.0 (api 11.0) moves the first-token-anchored
    C_post to 5.0 + D'=2.0 = 7.0 (exactly the +3.0 inflation) while the pre-TTFT
    dispatch-anchored C_pre stays at 1.0. No cycle RuntimeError."""
    parsed = _build(_TTFT_TRACE)
    clock = VirtualClock()
    issuer = _TTFTIssuer(
        clock,
        api_by_id={_P: 11.0, _C_PRE: 1.0, _C_POST: 1.0, _TAIL: 0.5},
        ttft_by_id={_P: 5.0},
    )

    await _run_virtual(parsed, issuer)

    assert issuer.dispatched_at[_C_PRE] == pytest.approx(1.0, abs=1e-3)
    assert issuer.dispatched_at[_C_POST] == pytest.approx(7.0, abs=1e-3)


@pytest.mark.asyncio
async def test_parent_errors_before_first_token_falls_back_to_dispatch():
    """P errors after 0.5s without streaming a first token. ``_finalize_node``
    latches the event with NO wall entry, so C_post gates at the dispatch
    fallback P_dispatch(0.0) + D(4.0) = 4.0."""
    parsed = _build(_TTFT_TRACE)
    clock = VirtualClock()
    issuer = _ErrorBeforeFirstTokenIssuer(
        clock,
        api_by_id={_P: 8.0, _C_PRE: 1.0, _C_POST: 1.0, _TAIL: 0.5},
        ttft_by_id={_P: 2.0},
        error_node=_P,
        error_after=0.5,
    )

    await _run_virtual(parsed, issuer)

    assert issuer.dispatched_at[_C_POST] == pytest.approx(4.0, abs=1e-3)


@pytest.mark.asyncio
async def test_ignore_edge_delays_skips_the_first_token_wait(monkeypatch):
    """``AIPERF_GRAPH_IGNORE_EDGE_DELAYS`` short-circuits the first-token wait
    entirely: every node dispatches at its scheduling instant, exactly once, in
    causal order (P before both children)."""
    monkeypatch.setattr(Environment.GRAPH, "IGNORE_EDGE_DELAYS", True, raising=False)

    parsed = _build(_TTFT_TRACE)
    clock = VirtualClock()
    issuer = _TTFTIssuer(
        clock,
        api_by_id={_P: 8.0, _C_PRE: 1.0, _C_POST: 1.0, _TAIL: 0.5},
        ttft_by_id={_P: 2.0},
    )

    await _run_virtual(parsed, issuer)

    from collections import Counter

    counts = Counter(issuer.dispatched)
    assert set(counts) == {_P, _C_PRE, _C_POST, _TAIL}
    assert all(n == 1 for n in counts.values()), (
        f"every node must dispatch exactly once; got {counts}"
    )
    assert issuer.dispatched_at[_P] == pytest.approx(0.0, abs=1e-3)
    assert issuer.dispatched_at[_C_PRE] == pytest.approx(0.0, abs=1e-3)
    assert issuer.dispatched_at[_C_POST] == pytest.approx(0.0, abs=1e-3)
    order = issuer.dispatched
    assert order.index(_P) < order.index(_C_PRE)
    assert order.index(_P) < order.index(_C_POST)


@pytest.mark.asyncio
async def test_parent_resolves_without_first_token_falls_back():
    """A streaming parent whose issuer never streams a first token (no ttft)
    resolves normally at api=3.0; ``_finalize_node`` latches C_post's event and
    the gate falls back to P_dispatch(0.0) + D(4.0) = 4.0."""
    parsed = _build(_TTFT_TRACE)
    clock = VirtualClock()
    issuer = _TTFTIssuer(
        clock,
        api_by_id={_P: 3.0, _C_PRE: 1.0, _C_POST: 1.0, _TAIL: 0.5},
        ttft_by_id={},  # P never streams a first token
    )

    await _run_virtual(parsed, issuer)

    assert issuer.dispatched_at[_C_PRE] == pytest.approx(1.0, abs=1e-3)
    assert issuer.dispatched_at[_C_POST] == pytest.approx(4.0, abs=1e-3)


@pytest.mark.asyncio
async def test_duplicate_stamp_during_dispatch_is_noop():
    """Invoking ``first_token_cb`` twice at the first-token instant does not
    raise and does not shift C_post (recorded-speed observed anchor = 4.0)."""
    parsed = _build(_TTFT_TRACE)
    clock = VirtualClock()
    issuer = _DoubleStampTTFTIssuer(
        clock,
        api_by_id={_P: 8.0, _C_PRE: 1.0, _C_POST: 1.0, _TAIL: 0.5},
        ttft_by_id={_P: 2.0},
        double_node=_P,
    )

    await _run_virtual(parsed, issuer)

    assert issuer.dispatched_at[_C_POST] == pytest.approx(4.0, abs=1e-3)


def test_first_token_stamp_is_idempotent_and_latches():
    """Direct unit test of the stamp closure: the first invocation records the
    wall and sets the latch; a second invocation is a no-op -- it neither
    re-reads the clock nor overwrites the recorded wall."""
    from aiperf.graph.dispatch.llm import _make_first_token_stamp

    class _FakeCtx:
        def __init__(self) -> None:
            self.node_first_token_wall_us: dict[str, float] = {}
            self._events: dict[str, asyncio.Event] = {}

        def first_token_event(self, node_id: str) -> asyncio.Event:
            return self._events.setdefault(node_id, asyncio.Event())

    ctx = _FakeCtx()
    walls = iter([100.0, 999.0])
    stamp = _make_first_token_stamp(ctx, "n1", lambda: next(walls))

    stamp()
    assert ctx.node_first_token_wall_us["n1"] == 100.0
    assert ctx.first_token_event("n1").is_set()

    stamp()  # duplicate: guarded before the clock read, wall must not change
    assert ctx.node_first_token_wall_us["n1"] == 100.0
    assert next(walls) == 999.0  # proves the 2nd clock read was NOT consumed
