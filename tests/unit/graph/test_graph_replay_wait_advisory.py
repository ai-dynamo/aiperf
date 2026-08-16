# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Characterization of the no-duration long-park replay advisory.

``AgentGraphReplayStrategy._advise_if_long_replay_waits_without_duration``
warns that a count/session/bare graph run will park on recorded delays with no
console output. Three properties are pinned here because each one was wrong in
a shipped run against a real dynamo corpus:

* The scan covers EVERY executor firing gate, including
  ``StaticEdge.min_start_delay_us`` -- the leading START-relative offset. It was
  the one gate the scan missed while the node-level field it did scan is never
  stamped by any recorded-trace producer.
* The threshold and the reported number are the EFFECTIVE park (recorded delay
  divided by ``--replay-speedup``), not the recorded delay. ``_scale_timing``
  rescales a per-trace copy at dispatch and never touches ``self._parsed``, so
  reading the parsed value reported a 1135s park for an 18.9s one.
* The message does not call these delays idle gaps, and does not point at
  ``--trace-idle-gap-cap-seconds``: the active-interval idle warp collapses only
  stretches where the WHOLE trace was idle, so a delay spanning a concurrent
  long-running request is busy in the capture and is left intact BY DESIGN.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pytest import param

from aiperf.common.constants import MICROS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.agent_graph_replay import (
    _EDGE_DELAY_FIELDS,
    AgentGraphReplayStrategy,
)
from aiperf.timing.strategies.graph_warmup import GraphWarmupKind

# Comfortably above the 30s IDLE_GAP_NO_DURATION_WARN_SECONDS default.
_LONG_S = 1200.0


class _Issuer:
    async def issue_graph_credit(self, turn: Any) -> bool:
        return True

    def mark_graph_sending_complete(self) -> None: ...
    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...
    async def end_graph_trace(self, trace_id: str) -> None: ...


def _parsed(
    *,
    node_min_start_us: float | None = None,
    edge: StaticEdge | None = None,
) -> ParsedGraph:
    """One-trace corpus carrying at most one node gate and one edge."""
    graph = GraphRecord(
        nodes={
            "n": LlmNode(
                prompt=["hi"], output="out", min_start_delay_us=node_min_start_us
            )
        },
        edges=[edge] if edge is not None else [],
        state={},
    )
    return ParsedGraph(
        graph=graph, graphs={}, traces=[TraceRecord(id="t", graph_ref=None)]
    )


def _advise(
    parsed: ParsedGraph,
    *,
    replay_speedup: float | None = None,
    phase: CreditPhase = CreditPhase.PROFILING,
    traces: list[TraceRecord] | None = None,
    open_loop_strict: bool = False,
    burst_phase_starts: bool = False,
    **phase_kwargs: Any,
) -> list[str]:
    """Run the advisory and return the notices it emitted, resolved to strings.

    ``traces`` defaults to the whole parsed corpus, mirroring ``execute_phase``'s
    fallback when ``setup_phase`` selected nothing.
    """
    warmup_kind = (
        GraphWarmupKind.BOUNDARY_SNAPSHOT if phase == CreditPhase.WARMUP else None
    )
    strategy = AgentGraphReplayStrategy(
        config=CreditPhaseConfig(
            phase=phase, timing_mode=TimingMode.AGENT_GRAPH, **phase_kwargs
        ),
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        warmup_kind=warmup_kind,
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
        replay_speedup=replay_speedup,
        open_loop_strict=open_loop_strict,
        burst_phase_starts=burst_phase_starts,
    )
    emitted: list[str] = []
    strategy.notice = lambda message, *a, **k: emitted.append(  # type: ignore[method-assign]
        message() if callable(message) else message
    )
    strategy._advise_if_long_replay_waits_without_duration(
        list(parsed.traces) if traces is None else traces
    )
    return emitted


@pytest.mark.parametrize(
    "parsed",
    [
        param(
            _parsed(
                edge=StaticEdge(
                    source="START",
                    target="n",
                    min_start_delay_us=_LONG_S * MICROS_PER_SECOND,
                )
            ),
            id="edge_min_start_delay",
        ),
        param(
            _parsed(
                edge=StaticEdge(
                    source="a",
                    target="n",
                    delay_after_predecessor_us=_LONG_S * MICROS_PER_SECOND,
                )
            ),
            id="edge_delay_after_predecessor",
        ),
        param(
            _parsed(
                edge=StaticEdge(
                    source="a",
                    target="n",
                    delay_after_predecessor_start_us=_LONG_S * MICROS_PER_SECOND,
                )
            ),
            id="edge_delay_after_predecessor_start",
        ),
        param(
            _parsed(node_min_start_us=_LONG_S * MICROS_PER_SECOND),
            id="node_min_start_delay",
        ),
    ],
)  # fmt: skip
def test_every_executor_firing_gate_trips_the_advisory(parsed: ParsedGraph) -> None:
    """Each gate ``_compute_firing_gate_us`` can park on is scanned.

    ``StaticEdge.min_start_delay_us`` is the case that regressed: it is the only
    one a recorded dynamo trace stamps for a gap-started chain
    (``interval_order.build_interval_edges``), while the node-level field the
    scan already had is never stamped by any recorded-trace producer. The
    node-level case stays covered because the field is decodable schema that a
    hand-authored ``dag_jsonl`` graph can supply.
    """
    emitted = _advise(parsed)
    assert len(emitted) == 1
    assert f"{_LONG_S:.0f}s" in emitted[0]


def test_advisory_is_silent_for_a_wait_free_corpus() -> None:
    """No gates set -> nothing to advise about."""
    assert _advise(_parsed()) == []


def test_replay_speedup_scales_the_threshold_comparison() -> None:
    """A recorded wait that the speedup shrinks below the threshold is silent.

    1200s recorded at ``--replay-speedup 60`` is a 20s park -- under the 30s
    default -- so advising about it is noise. This is the case that fired
    spuriously: the pre-fix comparison used the recorded 1200s.
    """
    parsed = _parsed(
        edge=StaticEdge(
            source="a",
            target="n",
            delay_after_predecessor_us=_LONG_S * MICROS_PER_SECOND,
        )
    )
    assert _advise(parsed, replay_speedup=60.0) == []


def test_advisory_reports_the_effective_park_and_the_recorded_value() -> None:
    """At a speedup the headline number is the real park; recorded rides along."""
    parsed = _parsed(
        edge=StaticEdge(
            source="a",
            target="n",
            delay_after_predecessor_us=6000.0 * MICROS_PER_SECOND,
        )
    )
    emitted = _advise(parsed, replay_speedup=60.0)
    assert len(emitted) == 1
    # 6000s recorded / 60 = a 100s real park, still over the 30s threshold.
    assert "parks up to 100s" in emitted[0]
    assert "6000s recorded" in emitted[0]
    assert "/60 speedup" in emitted[0]


def test_advisory_omits_the_recorded_note_without_a_speedup() -> None:
    """At speedup 1 the recorded value IS the park, so there is nothing to add."""
    parsed = _parsed(
        edge=StaticEdge(
            source="a",
            target="n",
            delay_after_predecessor_us=_LONG_S * MICROS_PER_SECOND,
        )
    )
    emitted = _advise(parsed)
    assert len(emitted) == 1
    assert f"parks up to {_LONG_S:.0f}s between turns" in emitted[0]
    assert "recorded," not in emitted[0]


def test_advisory_does_not_blame_the_idle_gap_cap() -> None:
    """The message must not send the operator after ``--trace-idle-gap-cap-seconds``.

    These delays are predecessor-to-successor waits, and the largest survivors on
    a real corpus span a concurrent long-running request -- busy in the
    recording, so the active-interval warp leaves them alone by design. Calling
    them idle gaps cost an operator an hour chasing a cap that was working.
    """
    parsed = _parsed(
        edge=StaticEdge(
            source="a",
            target="n",
            delay_after_predecessor_us=_LONG_S * MICROS_PER_SECOND,
        )
    )
    message = _advise(parsed)[0]
    assert "idle-gap corpus" not in message
    assert "inter-turn gap" not in message
    assert "predecessor-to-successor delay" in message
    # It may NAME the cap, but only to say the cap is not the lever.
    assert "deliberately leaves it intact" in message


def test_edge_delay_fields_covers_every_declared_delay_on_static_edge() -> None:
    """``_EDGE_DELAY_FIELDS`` must not drift from the ``StaticEdge`` schema.

    Mechanical, because this is THE bug that started here: ``min_start_delay_us``
    was a real firing gate that the advisory's hand-written field list omitted,
    so a START-rooted leading offset went unreported. A new delay field added to
    the struct would repeat that silently. Every ``*_delay*_us`` field on the
    struct must appear in the constant; a genuinely non-gating one has to be
    excluded here deliberately, in the open.
    """
    # Every microsecond-valued field, NOT just ones spelled "delay": a gate named
    # min_start_gate_us or hold_until_us would slip a "delay"-only filter and
    # reintroduce the omission this guard exists to stop.
    declared = {f for f in StaticEdge.__struct_fields__ if f.endswith("_us")}
    assert declared == set(_EDGE_DELAY_FIELDS), (
        "StaticEdge delay fields and _EDGE_DELAY_FIELDS disagree; the replay-wait "
        "advisory silently ignores any gate missing from the constant"
    )


def _multi_graph(delays_s: dict[str, float]) -> ParsedGraph:
    """One graph per trace id, each with a single long edge delay."""
    graphs = {
        tid: GraphRecord(
            nodes={"n": LlmNode(prompt=["hi"], output="out")},
            edges=[
                StaticEdge(
                    source="a",
                    target="n",
                    delay_after_predecessor_us=delay_s * MICROS_PER_SECOND,
                )
            ],
            state={},
        )
        for tid, delay_s in delays_s.items()
    }
    return ParsedGraph(
        graph=GraphRecord(nodes={}, edges=[], state={}),
        graphs=graphs,
        traces=[TraceRecord(id=tid, graph_ref=tid) for tid in delays_s],
    )


def test_advisory_scans_only_the_admitted_traces() -> None:
    """A wait in a trace the corpus selection excluded must not be reported.

    ``setup_phase`` bounds the corpus (``--num-dataset-entries``), so scanning
    the whole parse would advise about a park from a graph that never runs.
    """
    parsed = _multi_graph({"kept": 100.0, "dropped": 9000.0})
    kept = [t for t in parsed.traces if t.id == "kept"]

    assert "100s" in _advise(parsed, traces=kept)[0]
    # Sanity: the excluded graph really is the larger one.
    assert "9000s" in _advise(parsed)[0]


def test_advisory_deduplicates_a_shared_graph() -> None:
    """Many traces pointing at one GraphRecord are scanned once, not N times."""
    parsed = _parsed(
        edge=StaticEdge(
            source="a",
            target="n",
            delay_after_predecessor_us=_LONG_S * MICROS_PER_SECOND,
        )
    )
    many = [TraceRecord(id=f"t{i}", graph_ref=None) for i in range(50)]
    emitted = _advise(parsed, traces=many)
    assert len(emitted) == 1
    assert f"{_LONG_S:.0f}s" in emitted[0]


def _timestamped(starts_ms: list[int]) -> ParsedGraph:
    """Graph whose nodes carry absolute recorded starts and NO edge delays."""
    graph = GraphRecord(
        nodes={
            f"n{i}": LlmNode(
                prompt=["hi"], output=f"out{i}", recorded_start_unix_ms=start
            )
            for i, start in enumerate(starts_ms)
        },
        edges=[],
        state={},
    )
    return ParsedGraph(
        graph=graph, graphs={}, traces=[TraceRecord(id="t", graph_ref=None)]
    )


def test_open_loop_strict_measures_consecutive_recorded_starts() -> None:
    """Strict mode parks on absolute schedule offsets, not on edge delays.

    ``_strict_schedule_projection`` throws every edge away and re-roots each node
    at START on its own ``recorded_start_unix_ms - trace_zero`` offset, so the
    silent stretch is the gap between consecutive scheduled starts. Scanning the
    (discarded) edge fields would report 0s for this corpus.
    """
    base = 1_700_000_000_000
    # Gaps of 10s then 300s; the 300s one is the stretch that goes quiet.
    parsed = _timestamped([base, base + 10_000, base + 310_000])

    assert _advise(parsed) == []  # no edge delays -> nothing to report
    emitted = _advise(parsed, open_loop_strict=True)
    assert len(emitted) == 1
    assert "parks up to 300s" in emitted[0]


def test_open_loop_strict_falls_back_to_edge_delays_without_timestamps() -> None:
    """A graph carrying no timestamps is not projected, so its edges still rule.

    ``_strict_schedule_projection`` returns the graph unchanged when no node has
    a ``recorded_start_unix_ms``; the scan must mirror that fallback rather than
    silently reporting nothing.
    """
    parsed = _parsed(
        edge=StaticEdge(
            source="a",
            target="n",
            delay_after_predecessor_us=_LONG_S * MICROS_PER_SECOND,
        )
    )
    emitted = _advise(parsed, open_loop_strict=True)
    assert len(emitted) == 1
    assert f"{_LONG_S:.0f}s" in emitted[0]


def test_advisory_is_silent_when_edge_delays_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AIPERF_GRAPH_IGNORE_EDGE_DELAYS`` means nothing can park.

    ``TraceExecutor._apply_firing_delay`` returns before computing ANY gate when
    that flag is set, so the corpus can carry arbitrarily long delays and the run
    still parks for zero seconds. Advising about a wait that cannot happen is the
    same false positive as reporting an unscaled delay.
    """
    monkeypatch.setattr(Environment.GRAPH, "IGNORE_EDGE_DELAYS", True)
    parsed = _parsed(
        edge=StaticEdge(
            source="a",
            target="n",
            delay_after_predecessor_us=_LONG_S * MICROS_PER_SECOND,
        )
    )
    assert _advise(parsed) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        param({"phase": CreditPhase.WARMUP}, id="warmup_phase"),
        param({"expected_duration_sec": 10.0}, id="duration_set"),
    ],
)  # fmt: skip
def test_advisory_is_suppressed_when_it_does_not_apply(kwargs: dict[str, Any]) -> None:
    """Warmup never replays recorded waits; a duration already bounds the run."""
    parsed = _parsed(
        edge=StaticEdge(
            source="a",
            target="n",
            delay_after_predecessor_us=_LONG_S * MICROS_PER_SECOND,
        )
    )
    assert _advise(parsed, **kwargs) == []


def test_non_strict_open_loop_tolerates_an_untimestamped_corpus() -> None:
    """Without --open-loop-strict an untimestamped corpus is a legitimate run.

    It replays its AUTHORED edge delays -- the documented full replay -- so the
    new strict guard must not widen into refusing that.
    """
    parsed = _parsed(
        edge=StaticEdge(
            source="a",
            target="n",
            delay_after_predecessor_us=_LONG_S * MICROS_PER_SECOND,
        )
    )
    strategy = AgentGraphReplayStrategy(
        config=CreditPhaseConfig(
            phase=CreditPhase.PROFILING, timing_mode=TimingMode.AGENT_GRAPH
        ),
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
        open_loop_replay=True,
        open_loop_strict=False,
    )
    strategy._validate_recorded_starts(list(parsed.traces))  # must not raise


def test_burst_phase_starts_suppresses_leading_offset_reporting() -> None:
    """A leading offset burst will collapse must not be advised about.

    ``--burst-phase-starts`` zeroes START-sourced ``min_start_delay_us`` before
    the executor sees it, so a run that fires immediately was being told it
    "parks up to 1200s". This is the same false-positive class as reporting an
    unscaled delay, reintroduced by adding the leading offset to the scan.
    """
    parsed = _parsed(
        edge=StaticEdge(
            source="START", target="n", min_start_delay_us=_LONG_S * MICROS_PER_SECOND
        )
    )
    assert _advise(parsed) != []  # without burst it is a real park
    assert _advise(parsed, burst_phase_starts=True) == []


def test_burst_phase_starts_still_reports_inter_turn_delays() -> None:
    """Burst collapses only the LEADING offsets; mid-graph pacing survives."""
    parsed = _parsed(
        edge=StaticEdge(
            source="a",
            target="n",
            delay_after_predecessor_us=_LONG_S * MICROS_PER_SECOND,
        )
    )
    emitted = _advise(parsed, burst_phase_starts=True)
    assert len(emitted) == 1
    assert f"{_LONG_S:.0f}s" in emitted[0]


def _timestamped_multi(starts_by_trace: dict[str, list[int]]) -> ParsedGraph:
    """One graph PER TRACE, the shape the dynamo adapter actually emits."""
    graphs = {
        tid: GraphRecord(
            nodes={
                f"{tid}:{i}": LlmNode(
                    prompt=["hi"], output=f"{tid}:{i}_out", recorded_start_unix_ms=s
                )
                for i, s in enumerate(starts)
            },
            edges=[],
            state={},
        )
        for tid, starts in starts_by_trace.items()
    }
    return ParsedGraph(
        graph=GraphRecord(nodes={}, edges=[], state={}),
        graphs=graphs,
        traces=[TraceRecord(id=tid, graph_ref=tid) for tid in starts_by_trace],
    )


def test_strict_mode_pools_the_timeline_across_concurrent_traces() -> None:
    """Interleaved traces fill each other's gaps -- do not report the per-graph max.

    A: 0s, 200s. B: 100s, 300s. Each graph alone shows a 200s internal gap, but
    something fires every 100s, so the phase is never quiet longer than that.
    """
    base = 1_700_000_000_000
    parsed = _timestamped_multi(
        {"A": [base, base + 200_000], "B": [base + 100_000, base + 300_000]}
    )
    emitted = _advise(parsed, open_loop_strict=True)
    assert len(emitted) == 1
    assert "parks up to 100s" in emitted[0]


def test_strict_mode_counts_the_park_between_traces() -> None:
    """A gap that belongs to NO single graph is still a park.

    Two internally dense traces recorded an hour apart: per-graph maxima say 1s
    and the advisory stayed silent, while the run parks the full hour between
    them (``--open-loop-strict`` implies open-loop, so each trace is held to its
    own recorded start).
    """
    base = 1_700_000_000_000
    parsed = _timestamped_multi(
        {"A": [base, base + 1_000], "B": [base + 3_600_000, base + 3_601_000]}
    )
    emitted = _advise(parsed, open_loop_strict=True)
    assert len(emitted) == 1
    assert "parks up to 3599s" in emitted[0]


def test_execute_phase_hands_the_advisory_its_selected_corpus() -> None:
    """The scoping fix is only real if ``execute_phase`` passes the SELECTED list.

    The scan being trace-scoped is pinned above by hand-feeding ``traces``; that
    proves the helper, not the wiring. This pins the wiring: whatever
    ``setup_phase`` selected into ``_selected_traces`` is what the advisory is
    called with, so a corpus bounded by ``--num-dataset-entries`` cannot be
    advised about a trace it excluded.
    """
    parsed = _multi_graph({"kept": 100.0, "dropped": 9000.0})
    strategy = AgentGraphReplayStrategy(
        config=CreditPhaseConfig(
            phase=CreditPhase.PROFILING, timing_mode=TimingMode.AGENT_GRAPH
        ),
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
    )
    kept = [t for t in parsed.traces if t.id == "kept"]
    strategy._selected_traces = kept

    seen: list[list[TraceRecord]] = []
    strategy._advise_if_long_replay_waits_without_duration = (  # type: ignore[method-assign]
        lambda traces: seen.append(list(traces))
    )
    # Stop execute_phase right after the advisory call; the run itself is not
    # under test here.
    sentinel = RuntimeError("stop after advisory")

    async def _boom(_traces):
        raise sentinel

    strategy._run_traces_under_duration_budget = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="stop after advisory"):
        asyncio.run(strategy.execute_phase())

    assert seen == [kept]
