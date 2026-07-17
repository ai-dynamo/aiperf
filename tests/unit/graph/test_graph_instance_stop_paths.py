# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TC5 / TC7 / R3 — instance stop-path hygiene on ``GraphIRReplayStrategy``.

* TC5: a ``CreditIssueRefusedError`` unwinding an instance is a HEALTHY stop
  (request-count / duration gate closed): no ``errored_traces`` bump, no
  "unwound with error" warning; genuine errors keep the error path.
* TC7: the duration-timeout branch awaiting the cancelled lane-runner must NOT
  swallow an EXTERNAL cancellation of the strategy task itself.
* R3: ``--burst-phase-starts`` collapses the t*-relative leading offsets the
  snapshot chop stamps on START in-edges (the node-level field is never
  stamped by the trie producers), via the shared
  ``aiperf.graph.scheduler.collapse_leading_start_offsets``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.graph.analysis import trace_duration_us
from aiperf.graph.credit_dispatch_adapter import (
    CreditIssueRefusedError,
    GraphDispatchError,
)
from aiperf.timing.strategies.graph_ir_replay import (
    GraphIRReplayStrategy,
    _leaf_credit_refusal,
)

_FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


class _Config:
    """Minimal per-phase config stub (mirrors test_lane_fanout_recycle)."""

    timing_mode = None
    phase = None

    def __init__(self, **caps: Any) -> None:
        self.concurrency = caps.get("concurrency")
        self.expected_num_sessions = caps.get("expected_num_sessions")
        self.total_expected_requests = caps.get("total_expected_requests")
        self.expected_duration_sec = caps.get("expected_duration_sec")


def _strategy(
    parsed: Any,
    issuer: Any,
    *,
    ratio: float = 0.0,
    burst: bool = False,
    lifecycle: Any = None,
) -> GraphIRReplayStrategy:
    return GraphIRReplayStrategy(
        config=_Config(),
        credit_issuer=issuer,
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        lifecycle=lifecycle,
        start_min_ratio=ratio,
        start_max_ratio=ratio,
        burst_phase_starts=burst,
    )


class _SinkIssuer:
    """Base issuer stub satisfying the phase-completion surface."""

    def mark_graph_sending_complete(self) -> None: ...

    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...


# ---------------------------------------------------------------------------
# TC5 — issuer refusal is a clean stop
# ---------------------------------------------------------------------------


class _RefusingIssuer(_SinkIssuer):
    """Refuses every graph credit (the request-count / duration gate closed)."""

    async def issue_graph_credit(self, turn: Any) -> bool:
        return False


class _ExplodingIssuer(_SinkIssuer):
    """Raises a genuine (non-refusal) error on every issue."""

    async def issue_graph_credit(self, turn: Any) -> bool:
        raise RuntimeError("issuer exploded")


@pytest.mark.asyncio
async def test_issuer_refusal_is_clean_stop_not_error():
    """A refused instance completes without bumping ``errored_traces``."""
    parsed = from_weka_trace(str(_FIX))
    strategy = _strategy(parsed, _RefusingIssuer())

    await strategy.execute_phase()

    assert strategy.errored_traces == 0
    assert strategy.completed_traces == strategy.admitted_traces == 1


@pytest.mark.asyncio
async def test_genuine_error_still_counts_errored_trace():
    """Control: a non-refusal instance error keeps the existing error path."""
    parsed = from_weka_trace(str(_FIX))
    strategy = _strategy(parsed, _ExplodingIssuer())

    await strategy.execute_phase()

    assert strategy.errored_traces == 1
    assert strategy.completed_traces == 1


def test_leaf_credit_refusal_unwraps_groups_but_not_mixed():
    refusal = CreditIssueRefusedError("gate closed")
    assert _leaf_credit_refusal(refusal) is refusal

    grouped = ExceptionGroup("trace", [refusal])
    assert _leaf_credit_refusal(grouped) is refusal

    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [refusal])])
    assert isinstance(_leaf_credit_refusal(nested), CreditIssueRefusedError)

    mixed = ExceptionGroup("trace", [refusal, GraphDispatchError("real failure")])
    assert _leaf_credit_refusal(mixed) is None

    assert _leaf_credit_refusal(RuntimeError("other")) is None


# ---------------------------------------------------------------------------
# TC7 — external cancel not swallowed by the duration-timeout unwind
# ---------------------------------------------------------------------------


class _Lifecycle:
    def time_left_in_seconds(self) -> float:
        return 0.5


@pytest.mark.asyncio
async def test_external_cancel_reraises_during_duration_timeout_unwind(monkeypatch):
    """An external cancel landing on the ``await dispatch`` unwind re-raises.

    Simulates the reviewed race directly: the strategy's duration ``wait_for``
    reports a timeout while the lane runner is still unwinding, and the RUNNER
    then cancels the strategy task. Pre-fix, ``suppress(CancelledError)``
    swallowed that external cancel and the coroutine returned normally.
    """
    parsed = from_weka_trace(str(_FIX))
    strategy = _strategy(parsed, _SinkIssuer(), lifecycle=_Lifecycle())

    hang = asyncio.Event()

    async def _lanes_stub(traces: list[Any]) -> None:
        await hang.wait()

    strategy._run_lanes = _lanes_stub

    async def _fake_wait_for(fut: Any, timeout: float | None = None) -> None:
        # Duration budget "elapses" while the lane runner keeps running, so
        # the TimeoutError branch's ``await dispatch`` genuinely suspends.
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", _fake_wait_for)

    task = asyncio.get_running_loop().create_task(
        strategy._run_traces_under_duration_budget([])
    )
    await asyncio.sleep(0)  # park the task on ``await dispatch``
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()

    hang.set()  # release the orphaned lane-runner task
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# R3 — burst collapses START-edge leading offsets on chopped trie graphs
# ---------------------------------------------------------------------------


def _tstar_ratio_between(parsed: Any, a: str, b: str) -> float:
    arr = {nid: n.arrival_offset_us for nid, n in parsed.graph.nodes.items()}
    duration = trace_duration_us(parsed, parsed.traces[0])
    return ((arr[a] + arr[b]) / 2.0) / duration


def test_burst_phase_starts_zeroes_start_edge_leading_offsets():
    """Burst zeroes the chop's START-edge ``min_start_delay_us``; spread keeps it.

    The trie producers stamp leading offsets on START in-edges only (the
    node-level ``min_start_delay_us`` is never stamped), so the pre-fix
    node-only collapse made ``--burst-phase-starts`` a no-op on trie graphs.
    Inter-turn (non-START) edge delays must survive in both modes.
    """
    parsed = from_weka_trace(str(_FIX))
    trace = parsed.traces[0]
    ratio = _tstar_ratio_between(parsed, "trace_03_n3:0", "trace_03_n3:1")

    spread = _strategy(parsed, _SinkIssuer(), ratio=ratio, burst=False)
    burst = _strategy(parsed, _SinkIssuer(), ratio=ratio, burst=True)

    spread_graph, _ = spread._graph_at_t_star(trace, spread._plans[trace.id])
    burst_graph, _ = burst._graph_at_t_star(trace, burst._plans[trace.id])

    spread_starts = [e for e in spread_graph.graph.edges if e.source == "START"]
    burst_starts = [e for e in burst_graph.graph.edges if e.source == "START"]
    assert spread_starts and burst_starts

    # Spread keeps the t*-relative leading offset the chop stamped...
    assert any(e.min_start_delay_us for e in spread_starts)
    # ...burst collapses every leading offset to fire at phase-time 0.
    assert all(not e.min_start_delay_us for e in burst_starts)

    # Recorded inter-turn pacing (non-START completion edges) is untouched.
    burst_inner = [e for e in burst_graph.graph.edges if e.source != "START"]
    assert burst_inner
    assert any(e.delay_after_predecessor_us for e in burst_inner)
