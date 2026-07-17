# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix C — GraphIRReplayStrategy completion + accounting against the REAL bridge.

These tests wire the strategy to the production ``CreditIssuer`` +
``PhaseProgressTracker`` + ``CreditCallbackHandler`` (the actual event bridge the
``PhaseRunner`` waits on), with a fake ``CreditRouter`` that echoes each issued
``Credit`` back as a ``CreditReturn`` on a later tick (simulating the worker
round-trip). They prove the two PROVEN bugs are fixed:

* F1/A3 — no-cap completion: with NO request/session/duration cap the phase
  STILL completes -- ``execute_phase`` returns, ``all_credits_sent_event`` is
  set on executor-drain (the strategy owns it), and ``all_credits_returned_event``
  is set once every issued graph credit returns. (RED: the strategy never set
  the sent event, so a no-cap run hung on ``event.wait()`` forever.)
* F2/A2 — node-as-session: graph credits BYPASS ``CreditCounter`` session
  arithmetic; ``--num-conversations N`` runs N WHOLE traces (not N nodes) and
  the sent-count is NOT frozen after node 1 of trace 1. ``sent_sessions`` stays
  0 for graph credits.
* ``--request-count N`` caps total node dispatches at N then stops cleanly.

A direct ``CreditCounter`` unit test (no-trace path) guards that non-graph
session accounting is byte-for-byte unchanged.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]

_FIX_DIR = Path(__file__).parents[2] / "unit" / "graph" / "fixtures"
_MIN = _FIX_DIR / "weka_min.json"


def _parsed(path: Path):
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    return from_weka_trace(str(path))


async def _expected_dispatch_count(parsed) -> int:
    """Count the LLM/Tool dispatches the executor fires for the whole corpus."""
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


def _multi_trace_parsed(n: int):
    """Replicate the single-trace corpus into ``n`` distinct trace instances."""
    import msgspec

    base = _parsed(_MIN)
    t0 = base.traces[0]
    traces = [msgspec.structs.replace(t0, id=f"{t0.id}#{i}") for i in range(n)]
    return msgspec.structs.replace(base, traces=traces)


class _EchoRouter:
    """Fake CreditRouter: echoes each sent Credit back via the callback handler.

    ``send_credit`` schedules ``CreditReturn`` delivery to the real
    ``CreditCallbackHandler.on_credit_return`` on a later tick -- exactly the
    worker round-trip, exercising the production graph-return observer +
    counter + event bridge.
    """

    def __init__(self) -> None:
        self.handler: Any = None
        self.sent: list[Any] = []

    async def wait_for_workers(self, timeout: float) -> None:
        return None

    async def cancel_all_credits(self) -> None:
        return None

    async def send_credit(self, credit: Any) -> None:
        from aiperf.credit.messages import CreditReturn

        self.sent.append(credit)
        ret = CreditReturn(credit=credit, first_token_sent=True)
        asyncio.get_running_loop().call_soon(self._deliver, ret)

    def _deliver(self, ret: Any) -> None:
        asyncio.ensure_future(self.handler.on_credit_return("w0", ret))


def _build_real_stack(parsed, *, requests=None, sessions=None, duration=None):
    """Build the production issuer/progress/lifecycle/callback stack + strategy.

    Returns ``(strategy, issuer, progress, lifecycle, router)``.
    """
    from aiperf.common.enums import CreditPhase
    from aiperf.credit.callback_handler import CreditCallbackHandler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.plugin.enums import TimingMode as _TM
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.request_cancellation import (
        RequestCancellationConfig,
        RequestCancellationSimulator,
    )
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=_TM.GRAPH_IR,
        total_expected_requests=requests,
        expected_num_sessions=sessions,
        expected_duration_sec=duration,
    )
    lifecycle = PhaseLifecycle(config)
    progress = PhaseProgressTracker(config)
    stop_checker = StopConditionChecker(
        config=config, lifecycle=lifecycle, counter=progress.counter
    )
    concurrency = ConcurrencyManager()
    concurrency.configure_for_phase(CreditPhase.PROFILING, None, None)
    cancellation = RequestCancellationSimulator(RequestCancellationConfig())
    router = _EchoRouter()
    issuer = CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=stop_checker,
        progress=progress,
        concurrency_manager=concurrency,
        credit_router=router,
        cancellation_policy=cancellation,
        lifecycle=lifecycle,
    )
    handler = CreditCallbackHandler(concurrency)
    strategy = GraphIRReplayStrategy(
        config=config,
        conversation_source=None,
        scheduler=None,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=lifecycle,
        parsed_graph=parsed,
        register_observer=handler.set_graph_return_observer,
        max_concurrent_traces=8,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
    )
    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=progress,
        lifecycle=lifecycle,
        stop_checker=stop_checker,
        strategy=strategy,
    )
    router.handler = handler
    lifecycle.start()
    return strategy, issuer, progress, lifecycle, router


async def _drive(strategy, progress):
    """Run the strategy and then wait for the returned event, all under a guard.

    Mirrors the PhaseRunner: execute_phase sets all_credits_sent_event on drain,
    then we await all_credits_returned_event (set by the callback handler).
    """
    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=5.0)
    assert progress.all_credits_sent_event.is_set()
    await asyncio.wait_for(progress.all_credits_returned_event.wait(), timeout=5.0)


async def test_no_cap_run_completes_and_sets_both_events():
    """F1/A3: no request/session/duration cap -> phase still completes.

    Proves the strategy bridges executor-drain to ``all_credits_sent_event`` and
    that every issued graph credit returns (``all_credits_returned_event`` set).
    """
    parsed = _parsed(_MIN)
    expected = await _expected_dispatch_count(parsed)
    strategy, issuer, progress, lifecycle, router = _build_real_stack(parsed)

    await _drive(strategy, progress)

    assert len(router.sent) == expected
    assert progress.counter.requests_sent == expected
    assert progress.counter.final_requests_sent == expected
    # Node-as-session bug fixed: NO graph credit bumps session counters.
    assert progress.counter.sent_sessions == 0
    assert progress.counter.requests_completed == expected


async def test_num_conversations_runs_n_whole_traces():
    """F2/A2: --num-conversations N runs N WHOLE traces, not N nodes.

    The proven bug froze the sent count after node 1 of trace 1. Here N=2 over a
    4-trace corpus must dispatch exactly 2 traces' worth of nodes and complete.
    """
    parsed = _multi_trace_parsed(4)
    per_trace = await _expected_dispatch_count(_parsed(_MIN))
    strategy, issuer, progress, lifecycle, router = _build_real_stack(
        parsed, sessions=2
    )

    await _drive(strategy, progress)

    assert strategy.admitted_traces == 2
    assert strategy.completed_traces == 2
    # records == sum of the 2 admitted traces' node dispatches (NOT 2 nodes)
    assert len(router.sent) == per_trace * 2
    assert progress.counter.requests_sent == per_trace * 2
    assert progress.counter.final_requests_sent == per_trace * 2
    # NOT frozen after node 1: with per_trace == 3, the proven bug froze at 1.
    assert per_trace > 1
    assert progress.counter.final_requests_sent != 1
    assert progress.counter.sent_sessions == 0


async def test_num_conversations_unset_runs_whole_corpus():
    """Unset --num-conversations replays every trace in the corpus."""
    parsed = _multi_trace_parsed(3)
    per_trace = await _expected_dispatch_count(_parsed(_MIN))
    strategy, issuer, progress, lifecycle, router = _build_real_stack(parsed)

    await _drive(strategy, progress)

    assert strategy.admitted_traces == 3
    assert len(router.sent) == per_trace * 3


async def test_request_count_caps_node_dispatches_then_stops_cleanly():
    """--request-count N caps total node dispatches at N then stops (no hang).

    The issuer's RequestCountStopCondition gate refuses issuance once
    requests_sent >= N; ``issue_graph_credit`` returns False, which the adapter
    turns into a clean per-trace stop. The phase must NOT hang.
    """
    parsed = _multi_trace_parsed(4)
    per_trace = await _expected_dispatch_count(_parsed(_MIN))
    cap = per_trace + 1  # straddles trace 1/2 boundary
    strategy, issuer, progress, lifecycle, router = _build_real_stack(
        parsed, requests=cap
    )

    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=5.0)
    assert progress.all_credits_sent_event.is_set()
    await asyncio.wait_for(progress.all_credits_returned_event.wait(), timeout=5.0)

    # Exactly ``cap`` node dispatches were placed on the wire, no more.
    assert len(router.sent) == cap
    assert progress.counter.requests_sent == cap
    assert progress.counter.final_requests_sent == cap


async def test_drained_traces_release_all_per_trace_state():
    """A drained trace releases ALL of its per-trace state.

    This test PINS the invariant that the per-trace
    resource (the ``CreditDispatchAdapter`` and its parked-Future ``_waiters``)
    is FULLY released the moment a trace drains, and the phase's completion is
    driven purely by ``sent == returned`` -- no "ready-but-queued" style
    accounting term may hold
    a drained trace open. A regression that introduced such an
    accounting term would either leak adapters past drain or hang the phase.

    Parity-neutral: this is resource-cleanup correctness, NOT a wire/dispatch
    behavior change, so it touches no parity oracle axis.
    """
    parsed = _multi_trace_parsed(4)
    per_trace = await _expected_dispatch_count(_parsed(_MIN))
    strategy, issuer, progress, lifecycle, router = _build_real_stack(parsed)

    # Record every adapter the strategy builds so we can assert post-drain that
    # NONE retains an outstanding waiter (the per-tree ``outstanding`` analog).
    built: list[Any] = []
    original_build = strategy._build_adapter

    def _record(trace_id: str, instance_id: str, **kwargs: Any):
        adapter = original_build(trace_id, instance_id, **kwargs)
        built.append(adapter)
        return adapter

    strategy._build_adapter = _record  # type: ignore[method-assign]

    await _drive(strategy, progress)

    assert strategy.completed_traces == 4
    assert len(built) == 4
    # Every trace's adapter was removed from the live registry on drain: no
    # finished tree lingers (the "tree drained -> release slot" invariant).
    assert strategy._adapters == {}, (
        f"drained traces leaked adapters: {sorted(strategy._adapters)}"
    )
    # And no drained adapter still holds a parked dispatch Future -- every
    # outstanding request settled (returned/cancelled), the ONLY thing that
    # keeps a tree from draining now that the queued term is gone.
    for adapter in built:
        assert adapter.inflight_count == 0, (
            f"adapter {adapter._trace_id!r} retained "
            f"{adapter.inflight_count} outstanding waiter(s) past drain"
        )
    # Completion is sent == returned, no residual outstanding work.
    assert progress.counter.requests_sent == per_trace * 4
    assert progress.counter.requests_completed == per_trace * 4
    assert progress.counter.sent_sessions == 0
