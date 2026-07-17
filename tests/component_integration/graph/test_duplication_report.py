# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch duplication report (covers serial recycle).

Component-level: no real worker / ZMQ / mock server. A 5-trace corpus is
replayed at concurrency 5 with a session cap (10) that forces every lane to
recycle once, so ``total_instances_started`` (10) exceeds
``distinct_loaded_traces`` (5) and the duplication factor is 2.0.

Asserts the report contract:
- factor > 1 with cache-bust OFF emits a WARNING (clones collide on identical
  prefixes -- duplication without the antidote);
- the SAME duplicated run with cache-bust ON emits NO warning (per-instance
  markers make the duplication safe);
- the warning ALSO fires on the DURATION-cancel path (the most recycle-heavy
  mode: lanes recycle until the timer, so the lane future is cancelled -- the
  report must still emit, exactly once);
- the resolved ``endpoint.cache_bust`` reaches ``strategy._cache_bust`` through
  the REAL ``TimingConfig.from_run`` -> ``PhaseRunner._build_graph_ir_strategy``
  seam (guards the documented three-touch-wiring trap).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec
import pytest

from aiperf.common.enums import CacheBustTarget, CreditPhase

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]

_FIX_DIR = Path(__file__).parents[2] / "unit" / "graph" / "fixtures"
_MIN = _FIX_DIR / "weka_min.json"

_DISTINCT_TRACES = 5
_SESSION_CAP = 10  # 2x the corpus -> every lane recycles once (factor == 2.0)


@dataclass
class _EchoIssuer:
    """Fake CreditIssuer: each ``issue_graph_credit`` schedules an echoed return.

    Mirrors the harness in ``test_graph_ir_replay_strategy`` -- the echoed
    return is delivered on a later loop tick to the installed graph-return
    observer, simulating the worker round-trip without a worker.
    """

    observer: Any = None
    issued: int = 0
    returned: int = 0
    sending_complete_calls: int = 0
    all_returned_event_set: bool = False

    async def issue_graph_credit(self, turn: Any) -> bool:
        self.issued += 1
        asyncio.get_running_loop().call_soon(self._echo, turn)
        return True

    def _echo(self, turn: Any) -> None:
        self.returned += 1
        if self.observer is not None:
            self.observer(turn, None, False)

    def mark_graph_sending_complete(self) -> None:
        self.sending_complete_calls += 1

    def graph_all_returned(self) -> bool:
        return self.returned >= self.issued

    def set_graph_all_returned_event(self) -> None:
        self.all_returned_event_set = True


class _PhaseCfg:
    """Per-phase config stub carrying the fields the strategy reads.

    ``expected_num_sessions`` caps distinct roots ever started (the recycle
    gate), so a cap above the corpus size forces recycles -- serial duplication.
    ``sessions=None`` leaves recycle uncapped so a duration budget is the only
    bound (the duration-cancel scenario).
    """

    def __init__(self, *, concurrency: int, sessions: int | None) -> None:
        self.phase = CreditPhase.PROFILING
        self.concurrency = concurrency
        self.expected_num_sessions = sessions
        self.total_expected_requests = None
        self.expected_duration_sec = None


class _DurationLifecycle:
    """Lifecycle stub whose ``time_left_in_seconds`` reports a fixed budget.

    A float means ``--benchmark-duration`` is set: the duration stop condition
    bounds the phase and ``_run_traces_under_duration_budget`` wraps the lane
    dispatch in ``asyncio.wait_for(timeout=...)``, cancelling it when the budget
    elapses -- the recycle-heavy duration-cancel path.
    """

    def __init__(self, remaining: float) -> None:
        self._remaining = remaining

    def time_left_in_seconds(self, include_grace_period: bool = False) -> float:
        return self._remaining


def _five_trace_corpus(*, gap_free: bool = False):
    """Return a ParsedGraph whose ``traces`` holds 5 distinct-id instances.

    Replicates the single ``weka_min`` template with ``#N`` suffixes (the same
    id-replace pattern the multi-trace strategy test uses); every clone resolves
    to the base graph (``parsed.graphs`` is empty -> single-graph fallback) and
    keys back to the base template id (the worker strips everything after the
    first ``#``).

    ``gap_free`` zeros every edge delay so instances replay instantly -- used by
    the duration-cancel scenario, where a tiny real-time budget must let the
    lanes recycle (factor >> 1) BEFORE the timer cancels them, rather than
    parking on ``weka_min``'s recorded ~1s inter-turn idle gaps.
    """
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    base = from_weka_trace(str(_MIN))
    if gap_free:
        graph = base.graph
        zeroed_edges = [
            msgspec.structs.replace(
                e,
                **{
                    f: 0.0
                    for f in ("delay_after_predecessor_us", "min_start_delay_us")
                    if getattr(e, f, None) is not None
                },
            )
            for e in graph.edges
        ]
        base = msgspec.structs.replace(
            base, graph=msgspec.structs.replace(graph, edges=zeroed_edges)
        )
    t0 = base.traces[0]
    clones = [t0]
    clones.extend(
        msgspec.structs.replace(t0, id=f"{t0.id}#{i}")
        for i in range(1, _DISTINCT_TRACES)
    )
    return msgspec.structs.replace(base, traces=clones)


def _make_strategy(
    parsed,
    issuer: _EchoIssuer,
    cache_bust: CacheBustTarget,
    *,
    sessions: int | None = _SESSION_CAP,
    lifecycle: Any = None,
):
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    return GraphIRReplayStrategy(
        config=_PhaseCfg(concurrency=_DISTINCT_TRACES, sessions=sessions),
        conversation_source=None,
        scheduler=None,
        stop_checker=None,
        credit_issuer=issuer,
        lifecycle=lifecycle,
        parsed_graph=parsed,
        register_observer=lambda obs: setattr(issuer, "observer", obs),
        max_concurrent_traces=_DISTINCT_TRACES,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
        cache_bust=cache_bust,
    )


def _duplication_warnings(records: list[logging.LogRecord]) -> list[str]:
    return [
        r.getMessage()
        for r in records
        if r.levelno >= logging.WARNING and "duplication" in r.getMessage().lower()
    ]


async def test_cache_bust_target_threads_to_strategy():
    """The resolved cache-bust target lands on ``self._cache_bust`` (T11 reuses it)."""
    parsed = _five_trace_corpus()
    strategy = _make_strategy(
        parsed, _EchoIssuer(), cache_bust=CacheBustTarget.FIRST_TURN_PREFIX
    )
    assert strategy._cache_bust == CacheBustTarget.FIRST_TURN_PREFIX


async def test_duplication_warning_emitted_when_cache_bust_off(caplog):
    """factor > 1 with cache-bust OFF warns that clones collide on shared prefixes."""
    parsed = _five_trace_corpus()
    assert len(parsed.traces) == _DISTINCT_TRACES

    issuer = _EchoIssuer()
    strategy = _make_strategy(parsed, issuer, cache_bust=CacheBustTarget.NONE)
    await strategy.setup_phase()
    with caplog.at_level(logging.WARNING, logger="GraphIRReplayTiming"):
        # Generous ceiling: the phase ends on its own budget almost immediately
        # on an idle machine, but loaded Windows CI runners have been observed
        # to need well over 15s of wall clock end to end.
        await asyncio.wait_for(strategy.execute_phase(), timeout=60.0)

    # 5 lanes recycle up to the 10-session cap -> 10 instances over 5 traces.
    assert strategy._admitted_traces == _SESSION_CAP
    warnings = _duplication_warnings(caplog.records)
    assert warnings, "duplication warning must be emitted when cache-bust is OFF"
    assert any("cache-bust" in w.lower() for w in warnings)


async def test_duplication_warning_absent_when_cache_bust_on(caplog):
    """The SAME duplicated run with cache-bust ON emits NO duplication warning."""
    parsed = _five_trace_corpus()

    issuer = _EchoIssuer()
    strategy = _make_strategy(
        parsed, issuer, cache_bust=CacheBustTarget.FIRST_TURN_PREFIX
    )
    await strategy.setup_phase()
    with caplog.at_level(logging.WARNING, logger="GraphIRReplayTiming"):
        # Generous ceiling: the phase ends on its own budget almost immediately
        # on an idle machine, but loaded Windows CI runners have been observed
        # to need well over 15s of wall clock end to end.
        await asyncio.wait_for(strategy.execute_phase(), timeout=60.0)

    # Duplication still happened (same recycle dynamics)...
    assert strategy._admitted_traces == _SESSION_CAP
    # ...but the per-instance markers make it safe, so no warning fires.
    assert not _duplication_warnings(caplog.records), (
        "duplication warning must be suppressed when cache-bust is ON"
    )


async def test_duplication_warning_emitted_on_duration_cancel(caplog):
    """The report fires on the DURATION-cancel path -- the most recycle-heavy mode.

    No session cap: lanes recycle until the duration budget elapses, at which
    point ``_run_traces_under_duration_budget`` CANCELS the lane future. The
    report must still emit (from the caller's ``finally``), EXACTLY once, so the
    warning is not silently absent precisely where factor is largest.
    """
    parsed = _five_trace_corpus(gap_free=True)

    issuer = _EchoIssuer()
    strategy = _make_strategy(
        parsed,
        issuer,
        cache_bust=CacheBustTarget.NONE,
        sessions=None,
        lifecycle=_DurationLifecycle(0.1),
    )
    await strategy.setup_phase()
    with caplog.at_level(logging.WARNING, logger="GraphIRReplayTiming"):
        # Generous ceiling: the phase ends on its own budget almost immediately
        # on an idle machine, but loaded Windows CI runners have been observed
        # to need well over 15s of wall clock end to end.
        await asyncio.wait_for(strategy.execute_phase(), timeout=60.0)

    # Recycle-heavy: far more instances started than the 5-trace corpus (factor
    # >> 1), even though the lane future was cancelled by the timer.
    assert strategy._instances_started > _DISTINCT_TRACES
    warnings = _duplication_warnings(caplog.records)
    # Fires despite the cancel, and EXACTLY once (the caller's finally runs once).
    assert len(warnings) == 1, f"expected exactly one duplication warning: {warnings}"
    assert "cache-bust" in warnings[0].lower()


def _capturing_graph_runner(profiling_config):
    """A ``PhaseRunner`` wired for the real ``_build_graph_ir_strategy`` seam.

    Mirrors ``test_dispatch_plumbing``'s harness: ``PhaseRunner.__new__`` with the
    collaborators the graph branch reads, and a capturing ``GraphIRReplayStrategy``
    subclass so the forwarded kwargs are observable.
    """
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
    from aiperf.timing.graph_channel import GraphPhaseChannel
    from aiperf.timing.phase.runner import PhaseRunner
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    parsed = from_weka_trace(str(_MIN))
    captured: dict = {}

    class _CapturingStrategy(GraphIRReplayStrategy):
        def __init__(self, **kw):
            captured.update(kw)
            super().__init__(**kw)

    class _Handler:
        def set_graph_return_observer(self, obs) -> None:
            self._obs = obs

        def set_graph_first_token_observer(self, obs) -> None:
            self._ft_obs = obs

    channel = GraphPhaseChannel(parsed_graph=parsed)
    runner = PhaseRunner.__new__(PhaseRunner)
    runner._config = profiling_config
    runner._conversation_source = None
    runner._graph_channel = channel
    runner._scheduler = None
    runner._stop_checker = None
    runner._credit_issuer = object()
    runner._lifecycle = None
    runner._callback_handler = _Handler()
    return runner, captured, _CapturingStrategy


async def test_cache_bust_seam_reaches_strategy_end_to_end():
    """endpoint.cache_bust -> from_run -> runner -> strategy._cache_bust (3-touch seam).

    Drives the REAL wiring so a regression in ANY of the three touches
    (CreditPhaseConfig field, runner forwarding, strategy storage) is caught --
    this repo has a documented three-touch-wiring trap.
    """
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.plugin.enums import TimingMode
    from aiperf.timing.config import TimingConfig
    from tests.unit.conftest import make_run_from_cli

    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(_MIN),
        request_count=3,
        cache_bust=CacheBustTarget.FIRST_TURN_PREFIX,
    )
    run = make_run_from_cli(cfg)
    assert run.cfg.endpoint.cache_bust == CacheBustTarget.FIRST_TURN_PREFIX

    tc = TimingConfig.from_run(run)
    profiling = [p for p in tc.phase_configs if p.phase == CreditPhase.PROFILING]
    assert profiling, "expected a graph profiling phase"
    phase_cfg = profiling[0]
    assert phase_cfg.timing_mode == TimingMode.GRAPH_IR
    # Touch 1: from_run copies endpoint.cache_bust onto the CreditPhaseConfig.
    assert phase_cfg.cache_bust == CacheBustTarget.FIRST_TURN_PREFIX

    runner, captured, StrategyClass = _capturing_graph_runner(phase_cfg)
    strategy = runner._build_graph_ir_strategy(StrategyClass)
    # Touch 3: stored on the strategy (T11 reuses ``self._cache_bust``).
    assert strategy._cache_bust == CacheBustTarget.FIRST_TURN_PREFIX
    # Touch 2: forwarded as an explicit kwarg by the runner, not left to default.
    assert captured["cache_bust"] == CacheBustTarget.FIRST_TURN_PREFIX
