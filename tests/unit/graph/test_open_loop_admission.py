# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--concurrency`` bounds concurrent traces on the open-loop timestamped path.

The timestamped path opens one task per trace. Without admission control an
operator asking for concurrency 2 against a 5-trace corpus got 5 concurrent
executors. These tests pin the admission gate: at most ``concurrency`` trace
executors run at once, a released slot admits the next waiter, an unset
concurrency gates nothing, and admission NEVER re-anchors the schedule.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    TraceRecord,
)
from aiperf.graph.executor import TraceResult
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies import agent_graph_replay as agr
from aiperf.timing.strategies.agent_graph_replay import AgentGraphReplayStrategy
from aiperf.timing.strategies.graph_trace_planner import GraphTracePlanner


class _Issuer:
    async def issue_graph_credit(self, turn: Any) -> bool:
        return True

    def mark_graph_sending_complete(self) -> None: ...
    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...
    async def end_graph_trace(self, trace_id: str) -> None: ...


class _FakeAdapter:
    """Minimal stand-in for ``CreditDispatchAdapter``'s registry contract."""

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self.inflight_count = 0


class _FakePlanner:
    """Planner stub: every trace lowers to the same empty graph.

    ``per_trace`` overrides the lowered graph for named traces, so a test can
    give one trace a real ``recorded_start_unix_ms``.
    """

    def __init__(
        self, parsed: ParsedGraph, per_trace: dict[str, ParsedGraph] | None = None
    ) -> None:
        self._parsed = parsed
        self._per_trace = per_trace or {}

    def plan_for_lane(self, trace: TraceRecord, lane_index: int) -> None:
        return None

    def graph_at_t_star(self, trace: TraceRecord, plan: Any, **kwargs: Any) -> tuple:
        return self._per_trace.get(trace.id, self._parsed), trace

    def draw_index(self, index: int, total: int) -> int:
        return index % total

    def lane_salted_t_star(self, trace: TraceRecord, lane_index: int) -> float:
        # t*=0 -> every trace is spawnable, the default full-replay disposition.
        return 0.0

    def _draw_is_shuffled(self) -> bool:
        return False

    # Real selection logic over the stubbed draw: setup_phase now bounds the
    # corpus through the planner, so the stub must carry it -- along with the
    # recorded-start ordering the unshuffled bound uses.
    select_corpus = GraphTracePlanner.select_corpus
    _temporal_order = GraphTracePlanner._temporal_order


def _strategy(**phase_kwargs: Any) -> AgentGraphReplayStrategy:
    """Build a strategy over a hand-made phase config.

    A ``concurrency=`` kwarg here stands for an operator passing
    ``--concurrency``, so it carries the provenance flag the production
    converter would stamp -- the value alone never means "explicit".
    """
    return _strategy_from_config(
        CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENT_GRAPH,
            concurrency_explicitly_set="concurrency" in phase_kwargs,
            **phase_kwargs,
        )
    )


def _strategy_from_config(
    config: CreditPhaseConfig, clock: Any | None = None
) -> AgentGraphReplayStrategy:
    parsed = ParsedGraph(graph=GraphRecord(nodes={}, edges=[], state={}), traces=[])
    strategy = AgentGraphReplayStrategy(
        config=config,
        **({} if clock is None else {"clock": clock}),
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
    )
    strategy._open_loop_replay = True
    strategy._planner = _FakePlanner(parsed)
    strategy._build_adapter = lambda trace_id, instance_id, **kw: _FakeAdapter(  # type: ignore[method-assign]
        instance_id
    )
    strategy._first_token_sources_for = lambda trace: frozenset()  # type: ignore[method-assign]
    strategy._node_identity_for = lambda trace: None  # type: ignore[method-assign]
    strategy._release_adapter_if_idle = lambda instance_id: None  # type: ignore[method-assign]
    return strategy


class _ConcurrencyProbe:
    """Records peak concurrent executor runs and gates their completion."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.started: list[str] = []
        self.release = asyncio.Event()
        self.hold = False

    def executor_factory(self) -> type:
        probe = self

        class _FakeExecutor:
            def __init__(self, parsed: ParsedGraph, **kwargs: Any) -> None:
                self._parsed = parsed

            async def run(self, run_trace: Any) -> TraceResult:
                probe.active += 1
                probe.peak = max(probe.peak, probe.active)
                probe.started.append(getattr(run_trace, "id", "?"))
                try:
                    if probe.hold:
                        await probe.release.wait()
                    else:
                        await asyncio.sleep(0)
                finally:
                    probe.active -= 1
                # The real executor always returns one, and the strategy folds
                # its tool durations into the run totals. A stub returning None
                # would make every trace look like an error.
                return TraceResult(trace_id=getattr(run_trace, "id", "?"), channels={})

        return _FakeExecutor


def _traces(count: int) -> list[TraceRecord]:
    return [TraceRecord(id=f"t-{i}") for i in range(count)]


@pytest.mark.asyncio
async def test_timestamped_traces_respect_concurrency_ceiling(monkeypatch) -> None:
    """Concurrency 2 over 5 due traces never runs more than 2 executors at once."""
    strategy = _strategy(concurrency=2)
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    await strategy._run_timestamped_traces(_traces(5))

    assert probe.peak == 2
    assert len(probe.started) == 5


@pytest.mark.asyncio
async def test_released_slot_admits_the_next_waiting_trace(monkeypatch) -> None:
    """While the first traces hold their slots, no further trace starts."""
    strategy = _strategy(concurrency=2)
    probe = _ConcurrencyProbe()
    probe.hold = True
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    task = asyncio.ensure_future(strategy._run_timestamped_traces(_traces(5)))
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(probe.started) == 2, "held slots must block the remaining traces"

    probe.release.set()
    await task
    assert len(probe.started) == 5
    assert probe.peak == 2


@pytest.mark.asyncio
async def test_no_concurrency_configured_gates_nothing(monkeypatch) -> None:
    """Unset --concurrency keeps today's behavior: every trace launches at once."""
    strategy = _strategy()
    probe = _ConcurrencyProbe()
    probe.hold = True
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    task = asyncio.ensure_future(strategy._run_timestamped_traces(_traces(5)))
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(probe.started) == 5

    probe.release.set()
    await task


@pytest.mark.asyncio
async def test_admission_runs_after_the_recorded_start_wait(monkeypatch) -> None:
    """Ordering: a trace waits out its recorded start BEFORE it asks for a slot."""
    strategy = _strategy(concurrency=2)
    strategy._schedule_zero_unix_ms = 1_000_000
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    order: list[str] = []
    real_wait = strategy._wait_for_recorded_start

    async def _tracking_wait(parsed: ParsedGraph) -> bool:
        order.append("start-wait")
        # Propagate the admitted/released verdict: swallowing it would read as
        # "admission closed" and abandon every instance before it dispatches.
        return await real_wait(parsed)

    strategy._wait_for_recorded_start = _tracking_wait  # type: ignore[method-assign]
    real_acquire = strategy._acquire_trace_slot

    async def _tracking_acquire() -> int:
        order.append("admit")
        return await real_acquire()

    strategy._acquire_trace_slot = _tracking_acquire  # type: ignore[method-assign]

    await strategy._run_timestamped_traces(_traces(5))

    assert strategy._schedule_zero_unix_ms == 1_000_000
    assert order.count("start-wait") == 5
    assert order.count("admit") == 5
    assert order[0] == "start-wait"
    for i in range(0, len(order) - 1, 2):
        assert order[i] == "start-wait" and order[i + 1] == "admit"


@pytest.mark.asyncio
async def test_admission_delay_does_not_move_the_recorded_start_target(
    monkeypatch,
) -> None:
    """Slip fidelity with a REAL recorded start under virtual time.

    ``t-late`` is due 5 virtual seconds after schedule zero, but ``t-early``
    holds the only slot until virtual t=20. The late trace must still resolve
    its recorded start at anchor + 5s (the schedule is immutable) and merely
    START late -- execution slips, the schedule does not.
    """
    from aiperf.common.clock import SimClock

    clock = SimClock()
    zero_ms = 1_000_000
    late_graph = ParsedGraph(
        graph=GraphRecord(
            nodes={
                "n": LlmNode(
                    prompt=["hi"],
                    output="out",
                    recorded_start_unix_ms=zero_ms + 5_000,
                )
            },
            edges=[],
            state={},
        ),
        traces=[],
    )
    strategy = _strategy_from_config(
        CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENT_GRAPH,
            concurrency=1,
            concurrency_explicitly_set=True,
        ),
        clock=clock,
    )
    strategy._schedule_zero_unix_ms = zero_ms
    strategy._planner = _FakePlanner(
        ParsedGraph(graph=GraphRecord(nodes={}, edges=[], state={}), traces=[]),
        per_trace={"t-late": late_graph},
    )
    probe = _ConcurrencyProbe()
    probe.hold = True
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    wait_returned: dict[str, float] = {}
    started_at: dict[str, float] = {}
    real_wait = strategy._wait_for_recorded_start

    def _loop_now() -> float:
        return clock.perf_ns() / 1e9

    async def _tracking_wait(parsed: ParsedGraph) -> bool:
        # Propagate the admitted/released verdict: swallowing it would read as
        # "admission closed" and abandon every instance before it dispatches.
        admitted = await real_wait(parsed)
        wait_returned[str(len(wait_returned))] = _loop_now()
        return admitted

    strategy._wait_for_recorded_start = _tracking_wait  # type: ignore[method-assign]
    real_acquire = strategy._acquire_trace_slot

    async def _tracking_acquire() -> int:
        slot = await real_acquire()
        started_at[str(len(started_at))] = _loop_now()
        return slot

    strategy._acquire_trace_slot = _tracking_acquire  # type: ignore[method-assign]

    traces = [TraceRecord(id="t-early"), TraceRecord(id="t-late")]
    task = asyncio.ensure_future(strategy._run_timestamped_traces(traces))
    # Let t-early take the only slot, then fast-forward past t-late's recorded
    # start so it is due but still gated.
    for _ in range(20):
        await asyncio.sleep(0)
    clock.advance_to(5 * 1_000_000_000)
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(probe.started) == 1, "the late trace must be held at the gate"
    clock.advance_to(20 * 1_000_000_000)
    probe.release.set()
    await task

    anchor = strategy._schedule_anchor
    assert anchor is not None
    assert strategy._schedule_zero_unix_ms == zero_ms, "the anchor input is immutable"
    # The late trace resolved its recorded start exactly 5s after the anchor...
    assert wait_returned["1"] == pytest.approx(anchor + 5.0, abs=1e-6)
    # ...and only THEN slipped, starting when the held slot was released at 20s.
    assert started_at["1"] == pytest.approx(20.0, abs=1e-6)


@pytest.mark.asyncio
async def test_lane_limit_ramp_paces_timestamped_admission(monkeypatch) -> None:
    """set_lane_limit (--concurrency-ramp-duration) gates timestamped traces too."""
    strategy = _strategy(concurrency=4)
    strategy.set_lane_limit(1)
    probe = _ConcurrencyProbe()
    probe.hold = True
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    task = asyncio.ensure_future(strategy._run_timestamped_traces(_traces(5)))
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(probe.started) == 1, "limit 1 admits exactly one trace"

    strategy.set_lane_limit(4)
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(probe.started) == 4, "raising the limit admits the parked traces"

    probe.release.set()
    await task
    assert len(probe.started) == 5
    assert probe.peak == 4


@pytest.mark.asyncio
async def test_release_wakes_one_waiter_not_the_whole_herd(monkeypatch) -> None:
    """Each release hands the slot to ONE waiter; parked traces do not re-scan.

    Wake-all costs O(N^2) scans over the corpus, and the timestamped path opens
    one task per trace -- 50k traces would burn ~10^9 scheduler wakeups in the
    latency-critical timing-manager loop.
    """
    strategy = _strategy(concurrency=1)
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    scans = 0
    real_scan = strategy._lowest_free_slot

    def _counting_scan() -> int:
        nonlocal scans
        scans += 1
        return real_scan()

    strategy._lowest_free_slot = _counting_scan  # type: ignore[method-assign]

    traces = _traces(8)
    await strategy._run_timestamped_traces(traces)

    assert len(probe.started) == 8
    # One scan per arrival plus at most one per handoff.
    assert scans <= 2 * len(traces)


@pytest.mark.asyncio
async def test_gate_bounds_live_adapters_and_admitted_count(monkeypatch) -> None:
    """Parked traces hold no adapter and are not counted as admitted."""
    strategy = _strategy(concurrency=2)
    probe = _ConcurrencyProbe()
    probe.hold = True
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    task = asyncio.ensure_future(strategy._run_timestamped_traces(_traces(50)))
    for _ in range(20):
        await asyncio.sleep(0)

    assert len(probe.started) == 2
    assert strategy._admitted_traces == 2, (
        "admitted must count RUNNING traces, not parked ones"
    )
    assert len(strategy._adapters) == 2, (
        "--concurrency must bound adapter allocation, not just executor starts"
    )

    probe.release.set()
    await task
    assert strategy._admitted_traces == 50


class _FailingPlanner(_FakePlanner):
    """Planner that raises during lowering for named traces (pre-admission failure)."""

    def __init__(self, parsed: ParsedGraph, fail_ids: set[str]) -> None:
        super().__init__(parsed)
        self._fail_ids = fail_ids

    def graph_at_t_star(self, trace: TraceRecord, plan: Any, **kwargs: Any) -> tuple:
        if trace.id in self._fail_ids:
            raise RuntimeError(f"lowering blew up for {trace.id}")
        return super().graph_at_t_star(trace, plan, **kwargs)


def _with_failures(strategy: AgentGraphReplayStrategy, fail_ids: set[str]) -> None:
    strategy._planner = _FailingPlanner(
        ParsedGraph(graph=GraphRecord(nodes={}, edges=[], state={}), traces=[]),
        fail_ids,
    )


@pytest.mark.asyncio
async def test_mixed_pre_admission_failures_do_not_hard_fail(monkeypatch) -> None:
    """51 of 100 traces fail before admission -> ERROR line, NOT a raise."""
    strategy = _strategy(concurrency=4)
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())
    _with_failures(strategy, {f"t-{i}" for i in range(51)})

    await strategy._run_timestamped_traces(_traces(100))

    assert strategy._errored_traces == 51
    assert len(probe.started) == 49
    # No raise: 49 traces did real work.
    strategy.report_trace_failures()
    # The reported ratio must stay sane (errors <= denominator).
    assert strategy._errored_traces <= strategy._finished_traces
    # In-flight can never go negative.
    assert strategy._admitted_traces - strategy._completed_traces == 0


@pytest.mark.asyncio
async def test_every_trace_failing_pre_admission_hard_fails(monkeypatch) -> None:
    """Nothing ran at all -> the wholly-broken-run gate must still trip."""
    from aiperf.common.exceptions import InvalidStateError

    strategy = _strategy(concurrency=4)
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())
    _with_failures(strategy, {f"t-{i}" for i in range(5)})

    await strategy._run_timestamped_traces(_traces(5))

    assert probe.started == []
    with pytest.raises(InvalidStateError, match="every graph trace failed"):
        strategy.report_trace_failures()


@pytest.mark.asyncio
async def test_every_trace_failing_post_admission_still_hard_fails(monkeypatch) -> None:
    """The original all-failed case (failure inside the executor) is unchanged."""
    from aiperf.common.exceptions import InvalidStateError

    strategy = _strategy(concurrency=4)

    class _ExplodingExecutor:
        def __init__(self, parsed: ParsedGraph, **kwargs: Any) -> None: ...

        async def run(self, run_trace: Any) -> None:
            raise RuntimeError("executor blew up")

    monkeypatch.setattr(agr, "TraceExecutor", _ExplodingExecutor)

    await strategy._run_timestamped_traces(_traces(5))

    assert strategy._errored_traces == 5
    with pytest.raises(InvalidStateError, match="every graph trace failed"):
        strategy.report_trace_failures()


@pytest.mark.asyncio
async def test_setup_phase_accepts_lane_flags_on_open_loop() -> None:
    """The lane knobs are honored, not rejected, on the open-loop path."""
    strategy = _strategy(
        concurrency=2,
        expected_num_sessions=5,
        concurrency_ramp_duration_sec=10.0,
    )
    await strategy.setup_phase()


@pytest.mark.asyncio
async def test_setup_phase_open_loop_skips_the_oversubscription_guard() -> None:
    """The timestamped path never clones a trace, so over-provisioning is benign."""
    strategy = _strategy(concurrency=50)
    # Timestamped: open-loop setup also validates recorded starts, and this test
    # is about the oversubscription guard alone.
    strategy._parsed = ParsedGraph(
        graph=GraphRecord(
            nodes={
                "n": LlmNode(prompt=["hi"], output="out", recorded_start_unix_ms=1_000)
            },
            edges=[],
            state={},
        ),
        traces=_traces(2),
    )
    await strategy.setup_phase()


@pytest.mark.asyncio
async def test_setup_phase_closed_loop_still_guards_oversubscription() -> None:
    """The lane path can clone to fill lanes, so its guard is unchanged."""
    from aiperf.common.enums import CacheBustTarget
    from aiperf.common.exceptions import ConfigurationError

    strategy = _strategy(concurrency=50)
    strategy._open_loop_replay = False
    strategy._cache_bust = CacheBustTarget.NONE
    strategy._parsed = ParsedGraph(
        graph=GraphRecord(nodes={}, edges=[], state={}), traces=_traces(2)
    )
    with pytest.raises(ConfigurationError):
        await strategy.setup_phase()


def _default_graph_profiling_config() -> CreditPhaseConfig:
    """Build the profiling ``CreditPhaseConfig`` a BARE graph run actually produces.

    Hand-built ``CreditPhaseConfig(...)`` leaves ``concurrency`` at ``None``, a
    state production cannot reach: the default profiling phase type is
    ``concurrency``, whose ``concurrency`` field defaults to ``1``. Routing
    through the CLI converter + ``_build_profiling_config`` is what pins the real
    shape, so an unset ``--concurrency`` cannot silently look explicit.
    """
    import pydantic

    from aiperf.config.flags import CLIConfig
    from aiperf.config.flags._converter_profiling import build_profiling
    from aiperf.config.phases import PhaseConfig
    from aiperf.timing.config import _build_profiling_config
    from aiperf.timing.request_cancellation import RequestCancellationConfig

    cli = CLIConfig(url="http://localhost:8000/v1", model_names=["m"])
    phase = pydantic.TypeAdapter(PhaseConfig).validate_python(
        {"name": "profiling", **build_profiling(cli)}
    )
    return _build_profiling_config(
        phase,
        default_cancellation=RequestCancellationConfig(),
        phase_index=0,
        profiling_index=0,
        is_graph=True,
    )


@pytest.mark.asyncio
async def test_bare_default_graph_run_does_not_serialize_traces(monkeypatch) -> None:
    """A bare ``aiperf profile`` graph run must NOT gate admission.

    The production default phase carries ``concurrency=1`` (inherited, not
    chosen). Treating that as an explicit ceiling serialized every default
    open-loop replay to one trace at a time -- each trace draining its full
    recorded idle gaps before the next started.
    """
    config = _default_graph_profiling_config()
    assert config.concurrency == 1, "production default carries an inherited 1"

    strategy = _strategy_from_config(config)
    probe = _ConcurrencyProbe()
    probe.hold = True
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    task = asyncio.ensure_future(strategy._run_timestamped_traces(_traces(5)))
    for _ in range(20):
        await asyncio.sleep(0)
    assert len(probe.started) == 5, (
        f"inherited concurrency must not gate admission; only "
        f"{len(probe.started)} of 5 traces started"
    )

    probe.release.set()
    await task


# ---------------------------------------------------------------------------
# Issuer-refusal release (stop-condition termination on the open-loop path)
# ---------------------------------------------------------------------------


def _graph_due_at(recorded_start_unix_ms: int) -> ParsedGraph:
    """A one-node graph whose recorded start is ``recorded_start_unix_ms``."""
    return ParsedGraph(
        graph=GraphRecord(
            nodes={
                "n": LlmNode(
                    prompt=["hi"],
                    output="out",
                    recorded_start_unix_ms=recorded_start_unix_ms,
                )
            },
            edges=[],
            state={},
        ),
        traces=[],
    )


def _spread_corpus_strategy(
    monkeypatch, *, spacing_s: int = 10, count: int = 5
) -> tuple[AgentGraphReplayStrategy, list[TraceRecord], int, Any]:
    """Strategy over ``count`` traces recorded ``spacing_s`` apart.

    Returns ``(strategy, traces, last_due_s, clock)``. Sim time starts at 0 and
    each trace is due ``i * spacing_s`` seconds after schedule zero, so the
    clock reading at completion tells us exactly how much of the recorded
    timeline the phase actually sat through.
    """
    from aiperf.common.clock import SimClock

    zero_ms = 1_000_000
    clock = SimClock()
    strategy = _strategy_from_config(
        CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENT_GRAPH,
        ),
        clock=clock,
    )
    strategy._schedule_zero_unix_ms = zero_ms
    strategy._planner = _FakePlanner(
        ParsedGraph(graph=GraphRecord(nodes={}, edges=[], state={}), traces=[]),
        per_trace={
            f"t-{i}": _graph_due_at(zero_ms + i * spacing_s * 1_000)
            for i in range(count)
        },
    )
    traces = [TraceRecord(id=f"t-{i}") for i in range(count)]
    return strategy, traces, (count - 1) * spacing_s, clock


class _RefusingExecutor:
    """Executor stub whose every run trips the issuer's stop gate.

    Stands in for ``issue_graph_credit`` returning False once
    ``--request-count`` / ``--benchmark-duration`` / cancellation closes the
    gate: the adapter rejects the parked dispatch Future with
    ``CreditIssueRefusedError``, which unwinds out of ``TraceExecutor.run``.
    """

    runs = 0

    def __init__(self, parsed: ParsedGraph, **kwargs: Any) -> None: ...

    async def run(self, run_trace: Any) -> None:
        type(self).runs += 1
        raise agr.CreditIssueRefusedError("graph credit refused by issuer (stop gate)")


@pytest.mark.asyncio
async def test_issuer_refusal_releases_traces_parked_on_their_recorded_start(
    monkeypatch,
) -> None:
    """A closed stop gate must not leave the phase sitting out the whole timeline.

    ``--request-count N`` closes the issuer's stop gate for good (the counters
    it reads are monotonic), so every not-yet-dispatched trace is guaranteed to
    be refused. The traces are already parked inside
    ``_wait_for_recorded_start`` by then, so releasing them is what ends the
    phase -- a guard at the top of ``run_trace`` is a no-op, because every task
    is created before the first dispatch and is therefore already past it.

    Asserted on SIM TIME, not wall time: with the parked traces left to sleep,
    the phase advances the clock to the last trace's recorded start (40s here)
    even though the gate closed at 0s.
    """
    _RefusingExecutor.runs = 0
    strategy, traces, last_due_s, clock = _spread_corpus_strategy(monkeypatch)
    monkeypatch.setattr(agr, "TraceExecutor", _RefusingExecutor)

    task = asyncio.ensure_future(strategy._run_timestamped_traces(traces))
    await asyncio.gather(task, _pump_virtual_clock(clock, task))

    elapsed = clock.perf_ns() / 1e9
    assert elapsed == pytest.approx(0.0, abs=1e-6), (
        f"the stop gate closed at t=0 but the phase sat through {elapsed}s of the "
        f"{last_due_s}s recorded timeline; parked traces were never released"
    )
    assert _RefusingExecutor.runs == 1, (
        f"only the already-due trace should reach the executor; "
        f"{_RefusingExecutor.runs} did"
    )
    assert strategy._errored_traces == 0, "a stop-gate refusal is not a trace error"


@pytest.mark.asyncio
async def test_open_loop_pacing_survives_when_the_stop_gate_stays_open(
    monkeypatch,
) -> None:
    """Guard against over-fixing: no refusal => every recorded start is honored.

    The release path must trigger ONLY on a closed stop gate. With the gate
    open, the phase still walks the full recorded timeline -- that faithful
    pacing is the entire point of open-loop replay.
    """
    strategy, traces, last_due_s, clock = _spread_corpus_strategy(monkeypatch)
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    task = asyncio.ensure_future(strategy._run_timestamped_traces(traces))
    await asyncio.gather(task, _pump_virtual_clock(clock, task))

    elapsed = clock.perf_ns() / 1e9
    assert elapsed == pytest.approx(float(last_due_s), abs=1e-6), (
        f"open-loop pacing must still wait out the recorded timeline; "
        f"advanced only {elapsed}s of {last_due_s}s"
    )
    assert len(probe.started) == len(traces), "every trace must still run"


# ---------------------------------------------------------------------------
# The strategy paces on the injected Clock (production SimClock)
# ---------------------------------------------------------------------------


async def _pump_virtual_clock(clock: Any, task: asyncio.Future) -> None:
    """Driver pump: fast-forward sim time whenever the loop goes idle.

    The role ``clock.py`` documents for a virtual-time driver: poll to
    quiescence, advance to the next parked deadline, repeat.
    """
    while not task.done():
        for _ in range(10):
            await asyncio.sleep(0)
            if task.done():
                return
        next_ns = clock.next_event_time()
        if next_ns is not None:
            clock.advance_to(next_ns)


@pytest.mark.asyncio
async def test_strategy_paces_on_the_injected_clock(monkeypatch) -> None:
    """Open-loop pacing must run on the injected clock, not raw asyncio.

    ``clock.py`` exists so a multi-hour recorded replay can be validated in
    milliseconds under a ``SimClock``. ``TraceExecutor`` honors that; the
    strategy's own recorded-start pacing must too, or the open-loop timeline --
    the DEFAULT pacing mode and the dominant wall-time term -- stays
    wall-clock-bound and untestable without monkeypatching ``asyncio``.

    Deliberately patches nothing: the only time source is the injected clock.
    """
    from aiperf.common.clock import SimClock

    clock = SimClock()
    zero_ms = 1_000_000
    strategy = _strategy_from_config(
        CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENT_GRAPH,
        ),
        clock=clock,
    )
    strategy._schedule_zero_unix_ms = zero_ms
    strategy._planner = _FakePlanner(
        ParsedGraph(graph=GraphRecord(nodes={}, edges=[], state={}), traces=[]),
        per_trace={f"t-{i}": _graph_due_at(zero_ms + i * 10_000) for i in range(4)},
    )
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    traces = [TraceRecord(id=f"t-{i}") for i in range(4)]
    task = asyncio.ensure_future(strategy._run_timestamped_traces(traces))
    await asyncio.gather(task, _pump_virtual_clock(clock, task))

    assert len(probe.started) == 4, "every trace must still run"
    # 4 traces due 0/10/20/30s after schedule zero: sim time ends at 30s.
    assert clock.perf_ns() == 30 * 1_000_000_000, (
        f"pacing did not run on the injected clock (sim time {clock.perf_ns()}ns); "
        "the strategy is still reading asyncio's loop clock"
    )


# ---------------------------------------------------------------------------
# --allow-dataset-wrap is a PERMISSION, not an instruction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrap_permission_alone_does_not_duplicate_traffic(monkeypatch) -> None:
    """Wrap-allowed with NO stop condition still runs each trace exactly once.

    Pins what ``--allow-dataset-wrap`` actually does on the graph plane: it
    lets an over-subscribed ``--concurrency`` past the configuration guard, and
    nothing more. Reuse needs a stop condition to keep asking for work after
    the corpus is exhausted; without one the lanes make a single pass.

    Measured against the real CLI before this was written: a 3-trace corpus at
    ``--concurrency 5`` with the flag produced exactly 3 records, while adding
    ``--request-count 9`` produced 9 (a 3.00x duplication).
    """
    strategy = _strategy_from_config(
        CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.AGENT_GRAPH,
            concurrency=5,
            concurrency_explicitly_set=True,
            allow_dataset_wrap=True,
        )
    )
    strategy._open_loop_replay = False  # the only path that consults the flag
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(agr, "TraceExecutor", probe.executor_factory())

    traces = _traces(3)
    await strategy.setup_phase()
    await strategy._run_lanes(traces)

    assert len(probe.started) == 3, (
        "wrap permission alone must not replay any trace twice; "
        f"{len(probe.started)} instances ran over 3 traces"
    )
    assert sorted(probe.started) == sorted(t.id for t in traces)


@pytest.mark.asyncio
async def test_wrap_permission_is_inert_under_open_loop_replay() -> None:
    """Under the DEFAULT open-loop replay the flag changes nothing either way.

    ``_guard_explicit_oversubscription`` is reached only when
    ``open_loop_replay`` is False, so an over-subscribed concurrency is accepted
    with the flag set OR unset -- there are no lanes to fill.
    """
    for allow_wrap in (True, False, None):
        strategy = _strategy_from_config(
            CreditPhaseConfig(
                phase=CreditPhase.PROFILING,
                timing_mode=TimingMode.AGENT_GRAPH,
                concurrency=50,
                concurrency_explicitly_set=True,
                allow_dataset_wrap=allow_wrap,
            )
        )
        strategy._parsed = ParsedGraph(
            graph=GraphRecord(
                nodes={
                    "n": LlmNode(
                        prompt=["hi"], output="out", recorded_start_unix_ms=1_000
                    )
                },
                edges=[],
                state={},
            ),
            traces=_traces(2),
        )
        # Must not raise for any value of the flag.
        await strategy.setup_phase()
