# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for the extracted GraphTracePlanner.

These pin the planner's observable behavior so the extract-class refactor stays
verifiably behavior-preserving: identical seeds must yield identical draws, and
the default zero window must remain a byte-identical identity rewrite.
"""

from pathlib import Path

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.graph_trace_planner import (
    GraphTracePlanner,
    seed_for_draw_pass,
)

SEED = 42
DYNAMO_TRACE = (
    Path(__file__).resolve().parents[1]
    / "dataset"
    / "graph"
    / "adapters"
    / "fixtures"
    / "dynamo_nested"
    / "nested_2_level.jsonl.gz"
)


@pytest.fixture(scope="module")
def parsed():
    """The real dynamo nested trace, parsed once for the whole module."""
    return from_dynamo_trace(
        DYNAMO_TRACE, content_root_seed=SEED, content_tokenizer="builtin"
    )


def make_planner(parsed, strategy=None, seed=0):
    return GraphTracePlanner(
        parsed=parsed,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
        t_star_random_seed=seed,
        dataset_sampling_strategy=strategy,
    )


def test_sequential_draw_is_the_identity_cursor(parsed) -> None:
    """SEQUENTIAL/None must stay byte-for-byte the historical x % total draw."""
    planner = make_planner(parsed)
    assert [planner.draw_index(x, 4) for x in range(6)] == [0, 1, 2, 3, 0, 1]


def test_zero_total_draw_is_safe(parsed) -> None:
    assert make_planner(parsed).draw_index(3, 0) == 0


@pytest.mark.parametrize(
    "strategy",
    [
        param(DatasetSamplingStrategy.SHUFFLE, id="shuffle"),
        param(DatasetSamplingStrategy.RANDOM, id="random-coerced-to-shuffle"),
    ],
)  # fmt: skip
def test_shuffled_pass_covers_every_index_exactly_once(parsed, strategy) -> None:
    """Without-replacement coverage: one pass hits every index exactly once."""
    planner = make_planner(parsed, strategy, seed=42)
    assert sorted(planner.draw_index(x, 8) for x in range(8)) == list(range(8))


def test_same_seed_and_pass_yield_the_same_permutation(parsed) -> None:
    first = [
        make_planner(parsed, DatasetSamplingStrategy.SHUFFLE, seed=42).draw_index(x, 8)
        for x in range(8)
    ]
    second = [
        make_planner(parsed, DatasetSamplingStrategy.SHUFFLE, seed=42).draw_index(x, 8)
        for x in range(8)
    ]
    assert first == second


def test_distinct_passes_decorrelate(parsed) -> None:
    """A recycle pass re-permutes under a pass-salted seed."""
    planner = make_planner(parsed, DatasetSamplingStrategy.SHUFFLE, seed=42)
    first = [planner.draw_index(x, 8) for x in range(8)]
    second = [planner.draw_index(x, 8) for x in range(8, 16)]
    assert first != second


def test_seed_for_draw_pass_is_deterministic_and_pass_salted() -> None:
    assert seed_for_draw_pass(7, 0) == seed_for_draw_pass(7, 0)
    assert seed_for_draw_pass(7, 0) != seed_for_draw_pass(7, 1)
    assert seed_for_draw_pass(7, 0) != seed_for_draw_pass(8, 0)


def test_zero_window_yields_identity_rewrite(parsed) -> None:
    """Default ratios [0, 0] => t*=0 => full native replay, byte-identical."""
    planner = make_planner(parsed)
    trace = parsed.traces[0]
    plan = planner.plan_for_lane(trace, 0)
    run_parsed, run_trace = planner.graph_at_t_star(
        trace, plan, is_warmup=False, burst_phase_starts=False
    )

    assert run_trace is trace
    assert run_parsed.graph.nodes.keys() == parsed.graph.nodes.keys()


def _gap_started_parsed() -> ParsedGraph:
    """A t*=0 trie-shaped graph whose chain roots at START 24s in.

    Mirrors what ``interval_order.build_interval_edges`` emits for a node with no
    finished-before predecessor: a START in-edge carrying the warped arrival
    offset. The module fixture cannot stand in here -- its only START edge is at
    offset 0.0, so it would pass whether or not the collapse ran.
    """
    graph = GraphRecord(
        nodes={
            "a": LlmNode(
                prompt=["hi"], output="a_out", recorded_start_unix_ms=1_700_000_024_441
            ),
            "b": LlmNode(
                prompt=["hi"], output="b_out", recorded_start_unix_ms=1_700_000_029_441
            ),
        },
        edges=[
            StaticEdge(source="START", target="a", min_start_delay_us=24_441_000.0),
            StaticEdge(source="a", target="b", delay_after_predecessor_us=5_000_000.0),
        ],
        state={},
    )
    return ParsedGraph(
        graph=graph, graphs={}, traces=[TraceRecord(id="t", graph_ref=None)]
    )


def _leading_offsets(graph) -> list[float]:
    return [
        e.min_start_delay_us
        for e in graph.edges
        if e.source == "START" and e.min_start_delay_us
    ]


def test_zero_window_burst_collapses_leading_start_offsets() -> None:
    """``--burst-phase-starts`` must apply at t*=0, the DEFAULT disposition.

    A t*=0 trie graph already carries leading offsets: ``interval_order`` roots
    every gap-started chain at START with its warped arrival offset. The burst
    collapse used to be reachable only on the ``t*>0`` chop path, so passing the
    flag on an ordinary full replay silently did nothing -- the graph came back
    by identity with its offsets intact, and the run still parked 24s before its
    first turn.
    """
    parsed = _gap_started_parsed()
    planner = make_planner(parsed)
    trace = parsed.traces[0]
    plan = planner.plan_for_lane(trace, 0)

    plain, _ = planner.graph_at_t_star(
        trace, plan, is_warmup=False, burst_phase_starts=False
    )
    burst, _ = planner.graph_at_t_star(
        trace, plan, is_warmup=False, burst_phase_starts=True
    )

    # Guard: the graph must actually carry an offset, else this proves nothing.
    assert _leading_offsets(plain.graph) == [24_441_000.0]
    # Exact, not truthiness: `_leading_offsets` filters falsy values, so `== []`
    # alone would also pass if the collapse DELETED the edge or set it to None.
    # The executor gates on `node_firable + min_start_delay_us`, so the contract
    # is specifically that the edge survives carrying 0.0.
    burst_start_edges = [e for e in burst.graph.edges if e.source == "START"]
    assert [e.min_start_delay_us for e in burst_start_edges] == [0.0]
    # Inter-turn pacing is NOT part of the burst: only the leading offsets go.
    assert [e.delay_after_predecessor_us for e in burst.graph.edges] == [
        e.delay_after_predecessor_us for e in plain.graph.edges
    ]


def test_zero_window_without_burst_is_still_identity() -> None:
    """The default (no burst) path must stay a byte-identical passthrough."""
    parsed = _gap_started_parsed()
    planner = make_planner(parsed)
    trace = parsed.traces[0]
    run_parsed, _ = planner.graph_at_t_star(
        trace,
        planner.plan_for_lane(trace, 0),
        is_warmup=False,
        burst_phase_starts=False,
    )

    assert run_parsed.graph is parsed.graph


def test_zero_window_warmup_graph_is_empty(parsed) -> None:
    """t*<=0 primes nothing, so the warmup phase finalizes immediately."""
    planner = make_planner(parsed)
    trace = parsed.traces[0]
    run_parsed, _ = planner.graph_at_t_star(
        trace, planner.plan_for_lane(trace, 0), is_warmup=True, burst_phase_starts=False
    )

    assert run_parsed.graph.nodes == {}


def test_lane_zero_t_star_matches_the_prebuilt_plan(parsed) -> None:
    """lane_salted_t_star must equal plan_for_lane's t* -- same seed, same math."""
    planner = make_planner(parsed)
    trace = parsed.traces[0]
    plan = planner.plan_for_lane(trace, 0)

    assert planner.lane_salted_t_star(trace, 0) == plan.t_star_us


def test_plan_for_lane_is_cached(parsed) -> None:
    planner = make_planner(parsed)
    trace = parsed.traces[0]
    assert planner.plan_for_lane(trace, 0) is planner.plan_for_lane(trace, 0)
