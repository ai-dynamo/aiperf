# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test: drive the dataflow `TraceExecutor` over a real weka graph.

This is the first test on the branch that actually instantiates and runs
`TraceExecutor`. It ingests the weka fixture via `from_weka_trace`, then drives
the executor's real entrypoint (`TraceExecutor.run(trace)`) with a stub
credit issuer that records each `dispatch` call and returns a placeholder string
(downstream weka prompts do not read LLM output, so a placeholder is contractually
fine). Every LLM node must dispatch exactly once and the run must complete.
"""

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

import msgspec
import pytest

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
from aiperf.dataset.graph.adapters.weka.trie_build import (
    ReconCallbacks,
    build_trie_graph,
)
from aiperf.dataset.graph.models import (
    LlmNode,
    TraceRecord,
)
from aiperf.graph.executor import TraceExecutor

FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


class _RecordingCreditIssuer:
    """Stub credit issuer that records dispatch calls and returns a placeholder.

    Mirrors the contract that `dispatch/llm.py` relies on:
    `await issuer.dispatch(node, request, placement_ctx)`.
    Returns a `str` placeholder; downstream weka prompts do not read prior LLM
    output, so the value is never inspected by the runtime.
    """

    def __init__(self) -> None:
        self.dispatched: list[str] = []

    async def dispatch(
        self,
        node: Any,
        request: Any,
        ctx: Any,
        **kwargs: Any,
    ) -> str:
        self.dispatched.append(request.node_id)
        return f"placeholder::{request.node_id}"


@pytest.mark.asyncio
async def test_trace_executor_runs_weka_graph_end_to_end():
    # Real-content synthesis is always on; the synthetic fixture has hash_ids +
    # block_size > 0, so it synthesizes offline (gpt2 -> builtin tokenizer
    # fallback) without reaching a HuggingFace download.
    parsed = from_weka_trace(str(FIX))
    assert parsed.traces, "expected at least one trace"
    graph = parsed.graph if not parsed.graphs else next(iter(parsed.graphs.values()))

    expected_dispatch_ids = {
        nid for nid, n in graph.nodes.items() if isinstance(n, LlmNode)
    }
    assert expected_dispatch_ids, "expected at least one dispatchable LLM node"

    issuer = _RecordingCreditIssuer()
    executor = TraceExecutor(parsed, credit_issuer=issuer)

    async with asyncio.TaskGroup():
        for trace in parsed.traces:
            result = await executor.run(trace)
            assert result.trace_id == trace.id

    assert len(issuer.dispatched) == len(expected_dispatch_ids), (
        f"expected {len(expected_dispatch_ids)} dispatches "
        f"({sorted(expected_dispatch_ids)}), got {sorted(issuer.dispatched)}"
    )
    assert set(issuer.dispatched) == expected_dispatch_ids


class _OverflowOnceCreditIssuer:
    """Stub issuer that raises ``_NodeOverflowTerminate`` on the FIRST dispatch.

    Mirrors what ``CreditDispatchAdapter`` does when the worker returns a
    context-overflow error: the first dispatched LLM node terminates the
    trajectory early. Records every dispatch attempt so the test can assert the
    downstream turns never dispatch.
    """

    def __init__(self) -> None:
        self.dispatched: list[str] = []

    async def dispatch(
        self,
        node: Any,
        request: Any,
        ctx: Any,
        **kwargs: Any,
    ) -> str:
        from aiperf.graph.credit_dispatch_adapter import _NodeOverflowTerminate

        self.dispatched.append(request.node_id)
        if len(self.dispatched) == 1:
            raise _NodeOverflowTerminate("context_length_exceeded")
        return f"placeholder::{request.node_id}"


@pytest.mark.asyncio
async def test_overflow_terminates_trajectory_early_no_error():
    """When the first LLM node overflows, the executor stops dispatching the
    rest of the trace's turns and the run completes WITHOUT raising (the trace is
    not an errored trace).
    """
    parsed = from_weka_trace(str(FIX))
    graph = parsed.graph if not parsed.graphs else next(iter(parsed.graphs.values()))
    dispatchable = {nid for nid, n in graph.nodes.items() if isinstance(n, LlmNode)}
    # Sanity: the fixture is a linear chain of >1 dispatchable node so "stop the
    # rest" is observable.
    assert len(dispatchable) > 1

    issuer = _OverflowOnceCreditIssuer()
    executor = TraceExecutor(parsed, credit_issuer=issuer)

    # The run must NOT raise: the overflow is a clean early-termination, and the
    # downstream orphan cascade is swallowed (ctx.overflow_terminated).
    async with asyncio.TaskGroup():
        for trace in parsed.traces:
            result = await executor.run(trace)
            assert result.trace_id == trace.id

    # Exactly ONE dispatch happened (the overflowed node); every downstream turn
    # of the trajectory was suppressed.
    assert len(issuer.dispatched) == 1, (
        f"overflow must stop dispatching the rest of the trace; "
        f"got dispatches {issuer.dispatched}"
    )
    assert set(issuer.dispatched) < dispatchable


# --- AND-fan-in regression (T8-join) --------------------------------------

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


def _two_predecessor_trie_graph():
    """Build a trie graph whose resume turn AND-fans-in on MULTIPLE predecessors.

    Parent p0 (t=0) spawns two BLOCKING subagents; each subagent's inner turn
    derives from p0. The parent's resume turn p1 (t=4) starts after p0 AND both
    subagents finish. Under interval-order timing
    (:func:`_build_interval_edges`) p1's predecessors are the MAXIMAL
    finished-before frontier: the two subagent last turns. The content-parent p0
    (end 0.5) is transitively covered -- it finished before s1_inner started
    (0.5 <= 1.0), so ``p0 -> s1_inner -> p1`` drops p0 from p1's frontier. p1
    therefore declares TWO ``ChannelRequirement`` inputs (one per subagent
    ``_out``). This is the multi-predecessor early-fire the cycle guard tripped
    on.

    Returns ``(parsed_with_trace, p1_node_id, {predecessor node ids})``.
    """
    s1 = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "agent_1",
        "subagent_type": "Explore",
        "status": "completed",
        "duration_ms": 2000,
        "models": ["M"],
        "requests": [
            {"t": 1.0, "type": "n", "model": "M", "in": 128, "out": 16,
             "hash_ids": [50, 51], "api_time": 1.5},
        ],
    }  # fmt: skip
    s2 = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "agent_2",
        "subagent_type": "Explore",
        "status": "completed",
        "duration_ms": 2000,
        "models": ["M"],
        "requests": [
            {"t": 1.2, "type": "n", "model": "M", "in": 128, "out": 16,
             "hash_ids": [60, 61], "api_time": 1.5},
        ],
    }  # fmt: skip
    trace = WekaTrace.model_validate(
        {
            "id": "trie_join",
            "models": ["M"],
            "block_size": _BLOCK_SIZE,
            "hash_id_scope": "local",
            "requests": [
                {
                    "t": 0.0,
                    "type": "n",
                    "model": "M",
                    "in": 128,
                    "out": 64,
                    "hash_ids": [1, 2],
                    "api_time": 0.5,
                },
                s1,
                s2,
                {
                    "t": 4.0,
                    "type": "n",
                    "model": "M",
                    "in": 192,
                    "out": 32,
                    "hash_ids": [1, 2, 3],
                    "api_time": 1.0,
                },
            ],
        }
    )
    parsed, pool = build_trie_graph(trace, callbacks=_STUB_CALLBACKS)
    nodes = {nid: n for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)}
    by_offset = {n.arrival_offset_us: nid for nid, n in nodes.items()}
    p1 = by_offset[int(4.0 * 1e6)]
    # p1's interval-order frontier predecessors are the two subagent last turns;
    # the content-parent p0 (offset 0.0) is transitively covered and dropped.
    preds = {
        by_offset[int(1.0 * 1e6)],
        by_offset[int(1.2 * 1e6)],
    }

    # Precondition: p1 really has the multi-predecessor AND-join the bug fired
    # past (both subagent joins; p0 is frontier-dropped under interval-order).
    assert len(nodes[p1].inputs) == 2, nodes[p1].inputs
    assert {req.channel for req in nodes[p1].inputs} == {f"{pid}_out" for pid in preds}

    trie_trace = TraceRecord(id=trace.id)
    parsed = msgspec.structs.replace(parsed, traces=[trie_trace], segment_pool=pool)
    return parsed, p1, preds


@pytest.mark.asyncio
async def test_trie_and_fanin_join_node_fires_once_after_both_predecessors():
    """A 2-predecessor trie join runs with NO cycle error and fires the join ONCE.

    Before T8-join the join node declared no input channels, so the executor's
    ``await_inputs`` gate was a no-op: the node fired on its FIRST completing
    predecessor, finished, then the SECOND predecessor re-scheduled it ->
    ``RuntimeError("cycle detected: node ... re-scheduled after completing")``.

    With each node now reading its predecessors' ``{src}_out`` channels, the
    join WAITS for BOTH predecessors and fires exactly once. This is the
    EXECUTOR-level proof the static-structure tests miss (see
    gotcha_graph_adapter_tests_skip_validator).
    """
    parsed, join_id, pred_ids = _two_predecessor_trie_graph()
    all_ids = {nid for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)}

    issuer = _RecordingCreditIssuer()
    executor = TraceExecutor(parsed, credit_issuer=issuer)

    # The run must NOT raise the cycle RuntimeError (the regression). asyncio's
    # TaskGroup would wrap it in an ExceptionGroup; reaching the asserts at all
    # means no node re-scheduled after completing.
    async with asyncio.TaskGroup():
        for trace in parsed.traces:
            result = await executor.run(trace)
            assert result.trace_id == trace.id

    counts = Counter(issuer.dispatched)
    # Every node dispatched exactly once -- the join did not double-fire.
    assert set(counts) == all_ids
    assert counts[join_id] == 1, (
        f"join node {join_id!r} must fire EXACTLY once; got {counts[join_id]}"
    )
    for pid in pred_ids:
        assert counts[pid] == 1


@pytest.mark.asyncio
async def test_trie_and_fanin_join_waits_for_both_predecessors():
    """The join node fires only AFTER both predecessors have dispatched.

    Records dispatch ORDER and asserts the join id appears strictly after both
    predecessor ids -- the join's two ``count=1`` requirements force it to wait
    for both ``_out`` writes (each predecessor's LLM dispatch), not just the
    first.
    """
    parsed, join_id, pred_ids = _two_predecessor_trie_graph()

    issuer = _RecordingCreditIssuer()
    executor = TraceExecutor(parsed, credit_issuer=issuer)

    async with asyncio.TaskGroup():
        for trace in parsed.traces:
            await executor.run(trace)

    order = issuer.dispatched
    join_pos = order.index(join_id)
    for pid in pred_ids:
        assert pid in order[:join_pos], (
            f"join {join_id!r} dispatched before predecessor {pid!r}; order={order}"
        )


# --- virtual-time replay (Clock abstraction) ------------------------------


class _VTimeIssuer:
    """Records each node's VIRTUAL dispatch time, then consumes the node's
    recorded ``api_time`` in virtual time.

    ``api_by_id`` maps ``request.node_id`` -> recorded processing seconds. The
    executor records the node's finish as ``dispatch_start + api_time`` (the
    issuer's virtual sleep), so a successor's firing gate clears at the recorded
    end-to-start instant.
    """

    def __init__(self, clock: Any, api_by_id: dict[str, float]) -> None:
        self._clock = clock
        self._api = api_by_id
        self.dispatched_at: dict[str, float] = {}

    async def dispatch(
        self,
        node: Any,
        request: Any,
        ctx: Any,
        **kwargs: Any,
    ) -> str:
        nid = request.node_id
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


@pytest.mark.asyncio
async def test_trie_replays_recorded_timeline_on_virtual_time():
    """The REAL ``TraceExecutor`` on a ``VirtualClock`` reproduces the recorded
    per-node start times byte-exact, fast.

    The 2-predecessor trie has recorded starts p0=0, s1=1.0, s2=1.2, p1=4.0 with
    api_times 0.5/1.5/1.5/1.0 and no idle gaps, so the warped timeline equals the
    raw starts. Driven by a virtual clock pumped to each parked waiter, every
    node must dispatch at its recorded start -- proving the production firing
    loop (input gates, AND-join, edge-gate ``max``) reconstructs the timeline,
    not just a hand-rolled sim.
    """
    from aiperf.common.clock import VirtualClock

    parsed, _join_id, _pred_ids = _two_predecessor_trie_graph()
    nodes = {nid: n for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)}
    # recorded start (== raw t, no idle gap > cap) and api_time keyed by the
    # node id the executor dispatches (``request.node_id`` == graph node id).
    by_offset_us = {n.arrival_offset_us: nid for nid, n in nodes.items()}
    expected_start = {
        by_offset_us[0]: 0.0,
        by_offset_us[int(1.0 * 1e6)]: 1.0,
        by_offset_us[int(1.2 * 1e6)]: 1.2,
        by_offset_us[int(4.0 * 1e6)]: 4.0,
    }
    api_by_id = {
        by_offset_us[0]: 0.5,
        by_offset_us[int(1.0 * 1e6)]: 1.5,
        by_offset_us[int(1.2 * 1e6)]: 1.5,
        by_offset_us[int(4.0 * 1e6)]: 1.0,
    }

    clock = VirtualClock()
    issuer = _VTimeIssuer(clock, api_by_id)
    executor = TraceExecutor(parsed, credit_issuer=issuer, clock=clock)

    async def _phase() -> None:
        async with asyncio.TaskGroup():
            for trace in parsed.traces:
                await executor.run(trace)

    phase_task = asyncio.ensure_future(_phase())
    await _drive_virtual(clock, phase_task)

    assert set(issuer.dispatched_at) == set(expected_start)
    for nid, want in expected_start.items():
        assert issuer.dispatched_at[nid] == pytest.approx(want, abs=1e-3), (
            f"node {nid!r} dispatched at {issuer.dispatched_at[nid]:.4f}s, "
            f"expected recorded start {want:.4f}s"
        )
