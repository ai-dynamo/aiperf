# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--num-conversations N`` bounds the replayed graph corpus via the sampler.

On the open-loop graph replay path ``--num-conversations`` used to be accepted
and silently ignored: every loaded trace ran. These tests pin the SELECTION-time
bound -- N traces chosen through the existing dataset draw
(:meth:`GraphTracePlanner.draw_index`). The DEFAULT bounded draw is corpus
order -- a bound on recorded traffic is a temporal slice, so shuffling it would
destroy the arrival process -- and an explicit shuffle strategy still shuffles.
"""

from __future__ import annotations

from typing import Any

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    TraceRecord,
)
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
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
    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self.inflight_count = 0


class _ReversePlanner:
    """Planner stub whose draw order is the REVERSE of corpus order.

    Makes "went through the sampler" observable: a slicing implementation would
    return the corpus prefix, the sampler-routed one returns the tail.
    """

    def __init__(self, parsed: ParsedGraph) -> None:
        self._parsed = parsed

    def plan_for_lane(self, trace: TraceRecord, lane_index: int) -> None:
        return None

    def graph_at_t_star(self, trace: TraceRecord, plan: Any, **kwargs: Any) -> tuple:
        return self._parsed, trace

    def draw_index(self, index: int, total: int) -> int:
        return (total - 1) - (index % total)

    def _draw_is_shuffled(self) -> bool:
        # Force the non-shuffled branch so the stubbed reverse order selects.
        return False

    def _temporal_order(self, traces: list[TraceRecord]) -> list[int]:
        """Reverse corpus order, standing in for the real recorded-start sort.

        The unshuffled bound orders by recorded start, not by ``draw_index``,
        so the reversal has to live here for these tests to keep observing a
        selection that is NOT the corpus prefix.
        """
        return list(reversed(range(len(traces))))

    # Real selection logic over the stubbed order: exercises the production
    # code path while keeping the selected set observable.
    select_corpus = GraphTracePlanner.select_corpus


class _RecordingExecutor:
    """Records which traces actually ran."""

    ran: list[str] = []

    def __init__(self, parsed: ParsedGraph, **kwargs: Any) -> None: ...

    async def run(self, run_trace: Any) -> None:
        type(self).ran.append(getattr(run_trace, "id", "?"))


def _traces(count: int) -> list[TraceRecord]:
    return [TraceRecord(id=f"t-{i + 1}") for i in range(count)]


def _strategy(
    traces: list[TraceRecord], **phase_kwargs: Any
) -> AgentGraphReplayStrategy:
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENT_GRAPH,
        **phase_kwargs,
    )
    parsed = ParsedGraph(
        graph=GraphRecord(nodes={}, edges=[], state={}), traces=list(traces)
    )
    strategy = AgentGraphReplayStrategy(
        config=config,
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
    )
    strategy._open_loop_replay = True
    strategy._planner = _ReversePlanner(parsed)
    strategy._build_adapter = lambda trace_id, instance_id, **kw: _FakeAdapter(  # type: ignore[method-assign]
        instance_id
    )
    strategy._first_token_sources_for = lambda trace: frozenset()  # type: ignore[method-assign]
    strategy._node_identity_for = lambda trace: None  # type: ignore[method-assign]
    strategy._release_adapter_if_idle = lambda instance_id: None  # type: ignore[method-assign]
    return strategy


async def _run(strategy: AgentGraphReplayStrategy, monkeypatch) -> list[str]:
    _RecordingExecutor.ran = []
    monkeypatch.setattr(agr, "TraceExecutor", _RecordingExecutor)
    # setup_phase owns corpus selection; execute_phase replays what it chose.
    await strategy.setup_phase()
    await strategy.execute_phase()
    return list(_RecordingExecutor.ran)


@pytest.mark.asyncio
async def test_num_conversations_bounds_replayed_corpus(monkeypatch) -> None:
    """``--num-conversations 3`` over a 10-trace corpus replays exactly 3 traces."""
    strategy = _strategy(_traces(10), expected_num_sessions=3)

    ran = await _run(strategy, monkeypatch)

    assert len(ran) == 3
    assert len(set(ran)) == 3


@pytest.mark.asyncio
async def test_explicitly_non_sequential_planner_is_honored(
    monkeypatch,
) -> None:
    """A planner whose draw is not the corpus-order prefix is honored end to end."""
    traces = _traces(10)
    strategy = _strategy(traces, expected_num_sessions=3)

    ran = await _run(strategy, monkeypatch)

    prefix = [t.id for t in traces[:3]]
    assert ran != prefix
    assert set(ran) == {"t-10", "t-9", "t-8"}


@pytest.mark.asyncio
async def test_num_conversations_above_corpus_size_replays_whole_corpus(
    monkeypatch,
) -> None:
    """N > corpus size replays every trace exactly once (no cloning, no wrap)."""
    strategy = _strategy(_traces(4), expected_num_sessions=9)

    ran = await _run(strategy, monkeypatch)

    assert sorted(ran) == sorted(t.id for t in _traces(4))


@pytest.mark.asyncio
async def test_no_num_conversations_replays_whole_corpus(monkeypatch) -> None:
    """Unset ``--num-conversations`` leaves the corpus unbounded (unchanged)."""
    strategy = _strategy(_traces(5))

    ran = await _run(strategy, monkeypatch)

    assert len(ran) == 5


def _real_planner(
    seed: int,
    traces: list[TraceRecord],
    *,
    strategy: DatasetSamplingStrategy | None = None,
) -> GraphTracePlanner:
    """A REAL planner over ``traces`` -- no stubbing, so defaults are exercised."""
    parsed = ParsedGraph(
        graph=GraphRecord(nodes={}, edges=[], state={}), traces=list(traces)
    )
    return GraphTracePlanner(
        parsed=parsed,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
        t_star_random_seed=seed,
        dataset_sampling_strategy=strategy,
    )


def _planner(seed: int) -> GraphTracePlanner:
    """Planner with an EXPLICIT shuffle strategy (the opted-in draw)."""
    return _real_planner(seed, _traces(20), strategy=DatasetSamplingStrategy.SHUFFLE)


def test_select_corpus_is_seed_deterministic() -> None:
    """Same seed selects the same traces; a different seed selects differently."""
    traces = _traces(10)

    first = [t.id for t in _planner(42).select_corpus(traces, 3)]
    again = [t.id for t in _planner(42).select_corpus(traces, 3)]
    other = [t.id for t in _planner(1234).select_corpus(traces, 3)]

    assert first == again
    assert first != other


def test_select_corpus_shuffle_is_not_lexicographic_prefix() -> None:
    """A shuffled draw does not degenerate to the sorted-id prefix."""
    traces = _traces(20)

    selected = [t.id for t in _planner(7).select_corpus(traces, 5)]

    assert selected != [t.id for t in traces[:5]]
    assert len(set(selected)) == 5


def test_select_corpus_no_limit_returns_corpus_unchanged() -> None:
    """``limit`` of ``None`` / 0 / greater-than-size returns the whole corpus."""
    traces = _traces(4)
    planner = _planner(3)

    assert planner.select_corpus(traces, None) == traces
    assert planner.select_corpus(traces, 0) == traces
    assert sorted(t.id for t in planner.select_corpus(traces, 99)) == sorted(
        t.id for t in traces
    )


# --- Default-path selection (no explicit --dataset-sampling-strategy) ---------
#
# Bounding a corpus is a SUBSAMPLE: order fully determines WHICH traces
# represent the run. These exercise a REAL planner built the way resolution
# builds it for a user who never passed --dataset-sampling-strategy.


def test_default_sampling_bounded_selection_is_corpus_order_prefix() -> None:
    """Default (unset) strategy takes the first N in CORPUS ORDER.

    Pins the adjudicated default: a bound on a recorded trace corpus is a
    TEMPORAL subsample, so a shuffle would destroy the arrival process. If this
    ever flips back to shuffling, this test must fail loudly.
    """
    traces = _traces(20)

    selected = [t.id for t in _real_planner(42, traces).select_corpus(traces, 5)]

    assert selected == [t.id for t in traces[:5]]


def test_default_sampling_bounded_selection_is_seed_independent() -> None:
    """Default path is corpus order, so ``--random-seed`` cannot change it."""
    traces = _traces(20)

    first = [t.id for t in _real_planner(42, traces).select_corpus(traces, 5)]
    other = [t.id for t in _real_planner(1234, traces).select_corpus(traces, 5)]

    assert first == other == [t.id for t in traces[:5]]


def test_explicit_sequential_sampling_bounded_selection_is_prefix() -> None:
    """An EXPLICIT --dataset-sampling-strategy sequential matches the default."""
    traces = _traces(20)
    planner = _real_planner(42, traces, strategy=DatasetSamplingStrategy.SEQUENTIAL)

    selected = [t.id for t in planner.select_corpus(traces, 5)]

    assert selected == [t.id for t in traces[:5]]


def test_explicit_shuffle_bounded_selection_still_shuffles() -> None:
    """An EXPLICIT shuffle is still honored verbatim over the new default."""
    traces = _traces(20)
    planner = _real_planner(7, traces, strategy=DatasetSamplingStrategy.SHUFFLE)

    selected = [t.id for t in planner.select_corpus(traces, 5)]

    assert selected != [t.id for t in traces[:5]]
    assert len(set(selected)) == 5


@pytest.mark.parametrize(
    "strategy",
    [
        param(None, id="default-unset"),
        param(DatasetSamplingStrategy.SEQUENTIAL, id="explicit-sequential"),
        param(DatasetSamplingStrategy.SHUFFLE, id="explicit-shuffle"),
    ],
)  # fmt: skip
def test_unbounded_selection_returns_corpus_unchanged(
    strategy: DatasetSamplingStrategy | None,
) -> None:
    """Unbounded selection is byte-identical: the SAME list object, same order.

    The unbounded path and lane recycle must stay untouched under every
    strategy -- only a bound (limit < total) engages the subsample draw.
    """
    traces = _traces(6)
    planner = _real_planner(42, traces, strategy=strategy)

    assert planner.select_corpus(traces, None) is traces
    assert planner.select_corpus(traces, 0) is traces
    assert planner.select_corpus(traces, 6) is traces
    assert planner.select_corpus(traces, 99) is traces


# --- Schedule anchor over the SELECTED corpus --------------------------------


def _timestamped_parsed(starts: dict[str, int]) -> ParsedGraph:
    """Multi-graph corpus: one trace per id, each with its own recorded start."""
    graphs = {
        trace_id: GraphRecord(
            nodes={
                "n": LlmNode(
                    prompt=["hi"], output="out", recorded_start_unix_ms=start_ms
                )
            },
            edges=[],
            state={},
        )
        for trace_id, start_ms in starts.items()
    }
    return ParsedGraph(
        graph=GraphRecord(nodes={}, edges=[], state={}),
        graphs=graphs,
        traces=[TraceRecord(id=trace_id, graph_ref=trace_id) for trace_id in starts],
    )


@pytest.mark.asyncio
async def test_bounded_selection_anchors_on_selected_minimum(monkeypatch) -> None:
    """The schedule zero is the SELECTED corpus minimum, not the full-corpus one.

    Anchoring on the full corpus makes a bounded selection that excludes the
    earliest traces idle for ``selection_min - corpus_min`` before its first
    request -- long enough that a ``--benchmark-duration`` can expire having
    issued nothing.
    """
    zero_ms = 1_000_000
    starts = {
        "t-1": zero_ms,
        "t-2": zero_ms + 60_000,
        "t-3": zero_ms + 1_200_000,
        "t-4": zero_ms + 2_100_000,
        "t-5": zero_ms + 3_000_000,
    }
    parsed = _timestamped_parsed(starts)
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENT_GRAPH,
        expected_num_sessions=2,
    )
    strategy = AgentGraphReplayStrategy(
        config=config,
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
    )
    strategy._open_loop_replay = True
    assert strategy._schedule_zero_unix_ms == zero_ms

    # Reverse draw -> the two LATEST traces are selected.
    strategy._planner = _ReversePlanner(parsed)
    selected = strategy._select_replay_corpus(list(parsed.traces))

    assert [t.id for t in selected] == ["t-5", "t-4"]
    assert strategy._schedule_zero_unix_ms == starts["t-4"]


@pytest.mark.asyncio
async def test_bounded_selection_preserves_inter_trace_spacing(monkeypatch) -> None:
    """Re-anchoring shifts the whole selection equally: spacing is untouched."""
    zero_ms = 500_000
    starts = {
        "t-1": zero_ms,
        "t-2": zero_ms + 10_000,
        "t-3": zero_ms + 400_000,
        "t-4": zero_ms + 475_000,
    }
    parsed = _timestamped_parsed(starts)
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENT_GRAPH,
        expected_num_sessions=2,
    )
    strategy = AgentGraphReplayStrategy(
        config=config,
        credit_issuer=_Issuer(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
        register_first_token_observer=lambda obs: None,
        unregister_observer=lambda obs: None,
        unregister_first_token_observer=lambda obs: None,
    )
    strategy._open_loop_replay = True
    strategy._planner = _ReversePlanner(parsed)

    selected = strategy._select_replay_corpus(list(parsed.traces))
    zero = strategy._schedule_zero_unix_ms
    offsets = [starts[t.id] - zero for t in selected]

    # Earliest SELECTED trace fires immediately; spacing matches the recording.
    assert min(offsets) == 0
    assert max(offsets) - min(offsets) == starts["t-4"] - starts["t-3"]


# --- Bounded selection is a TEMPORAL slice, not a lexicographic one ----------
#
# Corpus order is ID order (the dynamo adapter id-sorts its traces), so a bare
# corpus-order prefix is a LEXICOGRAPHIC slice. That produces exactly the shape
# ``select_corpus`` documents itself as avoiding: N traces drawn from across a
# long capture are sparse arrivals separated by large idle gaps, nothing like
# the recorded load. Bounding must therefore slice the TIMELINE.


def _timed_corpus(
    specs: list[tuple[str, int | None]],
) -> tuple[ParsedGraph, list[TraceRecord]]:
    """Build a per-trace-graph corpus with the given recorded starts (unix ms).

    ``None`` gives a trace with no recorded start at all.
    """
    graphs = {
        trace_id: GraphRecord(
            nodes={
                "n": LlmNode(
                    prompt=["hi"], output="out", recorded_start_unix_ms=start_ms
                )
            },
            edges=[],
            state={},
        )
        for trace_id, start_ms in specs
    }
    traces = [TraceRecord(id=trace_id, graph_ref=trace_id) for trace_id, _ in specs]
    parsed = ParsedGraph(
        graph=GraphRecord(nodes={}, edges=[], state={}),
        traces=list(traces),
        graphs=graphs,
    )
    return parsed, traces


def _planner_over(parsed: ParsedGraph, seed: int = 42) -> GraphTracePlanner:
    return GraphTracePlanner(
        parsed=parsed,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
        t_star_random_seed=seed,
    )


def test_bounded_selection_slices_the_timeline_not_the_id_order() -> None:
    """The bound takes the EARLIEST N by recorded start, whatever the ids sort to.

    Ids here are deliberately reverse-ordered against time, so a corpus-order
    prefix would return the temporally LAST traces -- which is what the
    open-loop path did before this: ``--num-conversations 1`` on this corpus
    selected ``aaa_latest``.
    """
    base = 1_700_000_000_000
    # Listed in ID-SORTED order, the way the dynamo adapter emits its corpus
    # ("root-sorted" / "id-sorts the traces"), with time running the other way.
    # A corpus-order prefix therefore returns the LATEST traces.
    parsed, traces = _timed_corpus(
        [
            ("aaa_latest", base + 60_000),
            ("mmm_middle", base + 30_000),
            ("zzz_earliest", base + 0),
        ]
    )
    assert [t.id for t in traces] == sorted(t.id for t in traces), (
        "fixture must be id-sorted, or it cannot discriminate the two orders"
    )

    selected = [t.id for t in _planner_over(parsed).select_corpus(traces, 2)]

    assert selected == ["zzz_earliest", "mmm_middle"], (
        "bounding must slice the recorded timeline; got the id-order prefix"
    )


def test_bounded_selection_keeps_the_arrival_process_contiguous() -> None:
    """The selected traces are a CONTIGUOUS head of the capture, not a spread."""
    base = 1_700_000_000_000
    # Ids shuffled against time so corpus order cannot accidentally be right.
    parsed, traces = _timed_corpus(
        [
            ("d", base + 3_000),
            ("a", base + 9_000),
            ("c", base + 1_000),
            ("b", base + 6_000),
        ]
    )

    selected = [t.id for t in _planner_over(parsed).select_corpus(traces, 2)]

    assert selected == ["c", "d"], "expected the two earliest arrivals"


def test_bounded_selection_is_stable_for_equal_recorded_starts() -> None:
    """Traces sharing a recorded start keep corpus order (deterministic)."""
    base = 1_700_000_000_000
    parsed, traces = _timed_corpus([("first", base), ("second", base), ("third", base)])

    selected = [t.id for t in _planner_over(parsed).select_corpus(traces, 2)]

    assert selected == ["first", "second"]


def test_bounded_selection_prefers_timestamped_traces() -> None:
    """Untimestamped traces sort last, so a bound prefers the paceable ones.

    ``_validate_recorded_starts`` rejects a PARTIALLY timestamped open-loop
    corpus, and explicitly allows bounding onto a fully timestamped subset --
    so preferring timestamped traces turns a corpus that would be refused into
    a legitimate bounded run.
    """
    base = 1_700_000_000_000
    parsed, traces = _timed_corpus(
        [("untimed", None), ("late", base + 50_000), ("early", base)]
    )

    selected = [t.id for t in _planner_over(parsed).select_corpus(traces, 2)]

    assert selected == ["early", "late"]
