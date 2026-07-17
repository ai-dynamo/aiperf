# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R5 — GraphIRReplayStrategy drives a TraceExecutor per trace over a FAKE credit loop.

Component-level: no real worker, no ZMQ, no mock server. A fake ``CreditIssuer``
whose ``issue_graph_credit`` schedules an echoed ``CreditReturn`` delivered to the
strategy's installed graph-return observer (simulating the worker round-trip).

Asserts the corrected D2 contract:
- the phase COMPLETES (``execute_phase`` returns; no hang/deadlock);
- the number of graph credits issued equals the trace's executor LLM/Tool
  dispatch count;
- concurrent multi-trace runs all complete;
- a cancelled/errored return does NOT hang the phase.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]

_FIX_DIR = Path(__file__).parents[2] / "unit" / "graph" / "fixtures"
_MIN = _FIX_DIR / "weka_min.json"


async def _expected_dispatch_count(parsed) -> int:
    """Drive the executor with a recording stub to learn the LLM/Tool dispatch
    count for a trace (the executor decides firing/fan-out, so we count what it
    actually fires rather than re-deriving the topology)."""
    from aiperf.graph.executor import TraceExecutor

    class _Rec:
        def __init__(self) -> None:
            self.n = 0

        async def dispatch(self, node, request, ctx, **kw):
            self.n += 1
            return "x"

    rec = _Rec()
    ex = TraceExecutor(parsed, credit_issuer=rec)
    async with asyncio.TaskGroup():
        for trace in parsed.traces:
            await ex.run(trace)
    return rec.n


@dataclass
class _FakeReturn:
    """Drives the graph-return observer; carries (credit, error, cancelled)."""

    error: str | None = None
    cancelled: bool = False


@dataclass
class _EchoIssuer:
    """Fake CreditIssuer: each ``issue_graph_credit`` schedules an echoed return.

    The echoed ``CreditReturn`` is delivered on a later event-loop tick to the
    strategy's installed graph-return observer, keyed by the issued turn's
    ``(x_correlation_id, turn_index)`` -- exactly the correlation the real
    ``CreditCallbackHandler`` graph-return hook forwards. This simulates the
    worker round-trip without a worker.
    """

    observer: Any = None
    behavior: Any = None  # Callable[[turn], _FakeReturn] | None
    issued: int = 0
    returned: int = 0
    sent: list[Any] = field(default_factory=list)
    sending_complete_calls: int = 0
    all_returned_event_set: bool = False

    async def issue_graph_credit(self, turn: Any) -> bool:
        self.issued += 1
        self.sent.append(turn)
        ret = self.behavior(turn) if self.behavior is not None else _FakeReturn()
        asyncio.get_running_loop().call_soon(self._echo, turn, ret)
        return True

    def _echo(self, turn: Any, ret: _FakeReturn) -> None:
        self.returned += 1
        if self.observer is None:
            return
        self.observer(turn, ret.error, ret.cancelled)

    # New D2 completion contract: the strategy drives these once its executors
    # drain. The real ``CreditIssuer`` freezes counts + sets the phase events;
    # the fake just records that the strategy honored the contract.
    def mark_graph_sending_complete(self) -> None:
        self.sending_complete_calls += 1

    def graph_all_returned(self) -> bool:
        # All echoed returns are scheduled via call_soon; at the moment the
        # TaskGroup drains they have all been delivered (the executor awaited
        # each dispatch Future), so every issued credit has returned.
        return self.returned >= self.issued

    def set_graph_all_returned_event(self) -> None:
        self.all_returned_event_set = True


def _parsed(path: Path):
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    return from_weka_trace(str(path))


def _make_strategy(parsed, issuer: _EchoIssuer, **overrides):
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    # Pin the full-replay window so 0-arg call sites assert issued == full count.
    overrides.setdefault("start_min_ratio", 0.0)
    overrides.setdefault("start_max_ratio", 0.0)
    strategy = GraphIRReplayStrategy(
        config=overrides.pop("config", None),
        conversation_source=None,
        scheduler=None,
        stop_checker=None,
        credit_issuer=issuer,
        lifecycle=overrides.pop("lifecycle", None),
        parsed_graph=parsed,
        register_observer=lambda obs: setattr(issuer, "observer", obs),
        max_concurrent_traces=overrides.pop("max_concurrent_traces", 8),
        **overrides,
    )
    return strategy


async def test_strategy_completes_single_trace_and_issues_per_node_credits():
    parsed = _parsed(_MIN)
    expected = await _expected_dispatch_count(parsed)
    assert expected == 3

    issuer = _EchoIssuer()
    strategy = _make_strategy(parsed, issuer)
    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=5.0)

    assert issuer.issued == expected
    assert strategy.completed_traces == len(parsed.traces)
    # D2: the strategy OWNS completion -- it must signal sending-complete once
    # its executors drain (the PhaseRunner's all_credits_sent_event bridge).
    assert issuer.sending_complete_calls >= 1


async def test_strategy_completes_concurrent_multi_trace():
    # Two trace instances of the same template -> concurrent admission.
    import msgspec

    base = _parsed(_MIN)
    t0 = base.traces[0]
    t1 = msgspec.structs.replace(t0, id=t0.id + "#1")
    parsed = msgspec.structs.replace(base, traces=[t0, t1])
    per_trace = await _expected_dispatch_count(_parsed(_MIN))

    issuer = _EchoIssuer()
    strategy = _make_strategy(parsed, issuer, max_concurrent_traces=2)
    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=5.0)

    assert strategy.completed_traces == 2
    assert issuer.issued == per_trace * 2


async def test_strategy_does_not_hang_on_errored_return():
    parsed = _parsed(_MIN)
    # A dispatch error on a node is CONTAINED (mid-conversation resilience): the
    # executor does NOT unwind the trace's TaskGroup -- the failed request is
    # recorded as an error record, and the conversation continues past the
    # failed turn. So the trace COMPLETES (no hang) and is NOT counted as an
    # errored trace.
    issuer = _EchoIssuer(behavior=lambda turn: _FakeReturn(error="boom"))
    strategy = _make_strategy(parsed, issuer)
    await strategy.setup_phase()

    # Must NOT hang: the phase completes; the per-node failure is contained.
    await asyncio.wait_for(strategy.execute_phase(), timeout=5.0)
    assert strategy.completed_traces == len(parsed.traces)
    assert strategy.errored_traces == 0


async def test_strategy_does_not_hang_on_cancelled_return():
    parsed = _parsed(_MIN)
    issuer = _EchoIssuer(behavior=lambda turn: _FakeReturn(cancelled=True))
    strategy = _make_strategy(parsed, issuer)
    await strategy.setup_phase()

    # Cancelled return is contained the same way -- conversation continues, the
    # trace completes, no errored-trace unwind.
    await asyncio.wait_for(strategy.execute_phase(), timeout=5.0)
    assert strategy.completed_traces == len(parsed.traces)
    assert strategy.errored_traces == 0


class _Phase:
    """Minimal per-phase config stub carrying the phase + concurrency slots the
    strategy reads (``phase`` for the warmup/profiling t* disposition)."""

    def __init__(self, phase: Any) -> None:
        self.phase = phase
        self.concurrency = None
        self.expected_num_sessions = None


async def test_unset_concurrency_resolves_to_one():
    """An unset phase ``concurrency`` resolves to 1 (aiperf default), not 64.

    Regression for the removed ``AIPERF_GRAPH_MAX_CONCURRENT_TRACES`` fallback:
    with no explicit override and no phase concurrency, the trace-admission
    bound falls back to the plain aiperf default of 1.
    """
    from aiperf.common.enums import CreditPhase

    parsed = _parsed(_MIN)
    strategy = _make_strategy(
        parsed,
        _EchoIssuer(),
        config=_Phase(CreditPhase.PROFILING),
        max_concurrent_traces=None,
    )
    assert strategy._max_concurrent == 1


async def test_positive_window_profiling_phase_resumes_after_tstar():
    """PROFILING phase + the SAME t*=50% window replays the post-t* turns only
    (the warmup<->profiling boundary), strictly fewer than the full 3-turn replay."""
    from aiperf.common.enums import CreditPhase

    parsed = _parsed(_MIN)
    issuer = _EchoIssuer()
    strategy = _make_strategy(
        parsed,
        issuer,
        config=_Phase(CreditPhase.PROFILING),
        start_min_ratio=0.5,
        start_max_ratio=0.5,
        t_star_random_seed=42,
    )
    gt = next(iter(strategy._plans.values()))
    assert gt.t_star_us > 0

    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=5.0)

    full = await _expected_dispatch_count(parsed)
    assert full == 3
    # Post-t* turns at offsets 1.5s and 3.0s -> 2 profiled dispatches (< full).
    assert issuer.issued == 2, "profiling must replay only at/after-t* turns"
    assert 0 < issuer.issued < full
    assert strategy.completed_traces == 1


class _DurationLifecycle:
    """Lifecycle stub whose ``time_left_in_seconds`` mimics a configured budget.

    ``remaining`` is returned verbatim: a float means ``--benchmark-duration`` is
    set (the duration stop condition bounds the phase); ``None`` means no duration
    (count/session/bare mode), exactly the real ``PhaseLifecycle`` contract.
    """

    def __init__(self, remaining: float | None) -> None:
        self._remaining = remaining

    def time_left_in_seconds(self, include_grace_period: bool = False) -> float | None:
        return self._remaining


def _graph_with_max_gap(parsed, gap_us: float):
    """Return ``parsed`` with every static edge stamped with ``gap_us`` delay.

    Lets the idle-gap advisory be exercised at the unit level without the
    minute-scale integration fixture: ``_max_inter_turn_gap_seconds`` scans these
    ``delay_after_predecessor_us`` values.
    """
    import msgspec

    graph = parsed.graph
    new_edges = [
        msgspec.structs.replace(edge, delay_after_predecessor_us=gap_us)
        if hasattr(edge, "delay_after_predecessor_us")
        else edge
        for edge in graph.edges
    ]
    new_graph = msgspec.structs.replace(graph, edges=new_edges)
    return msgspec.structs.replace(parsed, graph=new_graph)


async def test_max_inter_turn_gap_seconds_reads_edge_and_node_delays():
    """``_max_inter_turn_gap_seconds`` reports the largest recorded gap (seconds)."""
    parsed = _graph_with_max_gap(_parsed(_MIN), 59_000_000.0)  # 59s in us
    strategy = _make_strategy(parsed, _EchoIssuer())
    assert strategy._max_inter_turn_gap_seconds() == pytest.approx(59.0)


async def test_max_inter_turn_gap_seconds_zero_for_gap_free_corpus():
    """A gap-free corpus (no edge/node delays) reports 0.0s -> advisory suppressed."""
    strategy = _make_strategy(_graph_with_max_gap(_parsed(_MIN), 0.0), _EchoIssuer())
    assert strategy._max_inter_turn_gap_seconds() == pytest.approx(0.0)


async def test_has_benchmark_duration_tracks_lifecycle_and_config():
    """``_has_benchmark_duration`` is True iff a duration budget is wired."""
    parsed = _parsed(_MIN)
    no_dur = _make_strategy(parsed, _EchoIssuer(), lifecycle=_DurationLifecycle(None))
    assert no_dur._has_benchmark_duration() is False

    with_dur = _make_strategy(parsed, _EchoIssuer(), lifecycle=_DurationLifecycle(10.0))
    assert with_dur._has_benchmark_duration() is True


async def test_advisory_fires_for_idle_gap_corpus_without_duration(caplog):
    """An idle-gap corpus with no duration emits the once-per-run NOTICE advisory."""
    import logging

    parsed = _graph_with_max_gap(_parsed(_MIN), 59_000_000.0)
    strategy = _make_strategy(parsed, _EchoIssuer(), lifecycle=_DurationLifecycle(None))
    with caplog.at_level(logging.INFO, logger="GraphIRReplayTiming"):
        strategy._advise_if_idle_gap_corpus_without_duration()
    assert any("--benchmark-duration" in r.getMessage() for r in caplog.records), (
        "idle-gap-without-duration advisory must mention --benchmark-duration"
    )


async def test_advisory_suppressed_when_duration_set(caplog):
    """A duration-bounded run never logs the advisory (the budget IS the bound)."""
    import logging

    parsed = _graph_with_max_gap(_parsed(_MIN), 59_000_000.0)
    strategy = _make_strategy(parsed, _EchoIssuer(), lifecycle=_DurationLifecycle(10.0))
    with caplog.at_level(logging.INFO, logger="GraphIRReplayTiming"):
        strategy._advise_if_idle_gap_corpus_without_duration()
    assert not any("--benchmark-duration" in r.getMessage() for r in caplog.records), (
        "advisory must be suppressed when a duration budget is configured"
    )


async def test_advisory_suppressed_for_gap_free_corpus(caplog):
    """A gap-free corpus (sub-threshold gaps) never logs the advisory."""
    import logging

    parsed = _graph_with_max_gap(_parsed(_MIN), 0.0)
    strategy = _make_strategy(parsed, _EchoIssuer(), lifecycle=_DurationLifecycle(None))
    with caplog.at_level(logging.INFO, logger="GraphIRReplayTiming"):
        strategy._advise_if_idle_gap_corpus_without_duration()
    assert not any("--benchmark-duration" in r.getMessage() for r in caplog.records), (
        "advisory must be suppressed for a gap-free corpus"
    )
