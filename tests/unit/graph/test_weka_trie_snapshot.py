# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``chop_trie_at_tstar`` t* snapshot chop on the trie ``ParsedGraph`` (Task 5).

The trie graph (Task 3) is the trivial ``LlmNode`` + ``StaticEdge`` realization
of a Weka trace. The t* chop drops every pre-``t*`` turn (warmed, not profiled)
and re-roots each surviving frontier turn from ``START`` at its t*-relative
offset, leaving each surviving node's ``prompt_segment_ids`` UNCHANGED (the full
pre-``t*`` prefix stays in the path so the worker still materializes the exact
resume prompt). ``t* <= 0`` returns the graph unchanged (full replay).

These tests build a small multi-turn trie graph with the same deterministic stub
callbacks the Task-3 build test uses (no tokenizer / corpus build) and assert the
chop STRUCTURE.
"""

from __future__ import annotations

from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
from aiperf.dataset.graph.adapters.weka.trie_build import (
    ReconCallbacks,
    build_trie_graph,
)
from aiperf.dataset.graph.models import (
    ChannelRequirement,
    LlmNode,
    ParsedGraph,
    StaticEdge,
)
from aiperf.dataset.graph.segment_ir.pool import SegmentPool
from aiperf.timing.snapshot_chop import chop_trie_at_tstar

BLOCK_SIZE = 64


def _stub_decode_block_tokens(hash_ids: list[int]) -> list[int]:
    out: list[int] = []
    for h in hash_ids:
        out.extend(range(h * 100, h * 100 + BLOCK_SIZE))
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


def _build(trace: WekaTrace) -> tuple[ParsedGraph, SegmentPool]:
    return build_trie_graph(trace, callbacks=_STUB_CALLBACKS)


def _llm_nodes(graph: ParsedGraph) -> dict[str, LlmNode]:
    return {nid: n for nid, n in graph.graph.nodes.items() if isinstance(n, LlmNode)}


def _edges(graph: ParsedGraph) -> list[StaticEdge]:
    return [e for e in graph.graph.edges if isinstance(e, StaticEdge)]


def _edge(edges: list[StaticEdge], source: str, target: str) -> StaticEdge | None:
    for e in edges:
        if e.source == source and e.target == target:
            return e
    return None


def _n(t: float, hashes: list[int], in_len: int, out: int, api_time: float,
       model: str = "M", type_: str = "n") -> dict:  # fmt: skip
    return {
        "t": t,
        "type": type_,
        "model": model,
        "in": in_len,
        "out": out,
        "hash_ids": hashes,
        "api_time": api_time,
    }


def _subagent(t: float, agent_id: str, requests: list[dict], status: str,
              duration_ms: int | None = None) -> dict:  # fmt: skip
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Explore",
        "status": status,
        "duration_ms": duration_ms,
        "requests": requests,
        "models": ["M"],
    }


def _trace(requests: list[dict]) -> WekaTrace:
    return WekaTrace.model_validate(
        {
            "id": "trace_0",
            "models": ["M"],
            "block_size": BLOCK_SIZE,
            "hash_id_scope": "local",
            "requests": requests,
        }
    )


def _join_trace() -> WekaTrace:
    """Parent p0 spawns subagents s1,s2; resume turn p1 AND-fans-in on both.

    Arrival offsets: p0=0 s, s1_inner=1 s, s2_inner=1.2 s, p1=4 s. Under
    interval-order timing p1 declares TWO ``inputs`` (``s1_inner_out``,
    ``s2_inner_out``); the content-parent p0 (end 0.5) is transitively covered
    (p0 -> s1_inner -> p1) and dropped from p1's finished-before frontier. Used
    to exercise the chop's input-rescoping when some of p1's predecessors are
    dropped.
    """
    s1 = _subagent(
        t=1.0,
        agent_id="agent_1",
        status="completed",
        duration_ms=2000,
        requests=[_n(t=1.0, hashes=[50, 51], in_len=128, out=16, api_time=1.5)],
    )
    s2 = _subagent(
        t=1.0,
        agent_id="agent_2",
        status="completed",
        duration_ms=2000,
        requests=[_n(t=1.2, hashes=[60, 61], in_len=128, out=16, api_time=1.5)],
    )
    return _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=0.5),
            s1,
            s2,
            _n(t=4.0, hashes=[1, 2, 3], in_len=192, out=32, api_time=1.0),
        ]
    )


def _four_turn_trace() -> WekaTrace:
    """Linear 4-turn conversation; each turn waits for its predecessor to finish.

    Turn k starts at t=k*10, runs api_time=1, so the inter-turn gap is
    ``k*10 - ((k-1)*10 + 1) = 9`` seconds. Arrival offsets are 0, 10, 20, 30 s.
    """
    return _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=1.0),
            _n(t=10.0, hashes=[1, 2, 3], in_len=192, out=32, api_time=1.0),
            _n(t=20.0, hashes=[1, 2, 3, 4], in_len=256, out=32, api_time=1.0),
            _n(t=30.0, hashes=[1, 2, 3, 4, 5], in_len=320, out=32, api_time=1.0),
        ]
    )


def _ordered_node_ids(graph: ParsedGraph) -> list[str]:
    """Node ids in ascending recorded arrival offset (turn order)."""
    nodes = _llm_nodes(graph)
    return sorted(nodes, key=lambda nid: nodes[nid].arrival_offset_us)


def test_chop_drops_pre_tstar_and_reroots_frontier() -> None:
    """t* between turn1 and turn2 drops turns 0,1; re-roots turn2 from START.

    With offsets [0, 10, 20, 30] s and t* = 25 s, turns 0,1,2 are pre-t* (dropped)
    and turn3 (offset 30 s) survives. turn3 re-roots from START with
    ``min_start_delay_us = (30 - 25) * 1e6``.
    """
    graph, _ = _build(_four_turn_trace())
    ids = _ordered_node_ids(graph)
    t_star_us = int(25.0 * 1e6)

    chopped = chop_trie_at_tstar(graph, t_star_us)

    surviving = _llm_nodes(chopped)
    # Only the offset-30 turn survives.
    assert set(surviving) == {ids[3]}
    # Re-rooted from START with the t*-relative arrival offset.
    edges = _edges(chopped)
    start_edge = _edge(edges, "START", ids[3])
    assert start_edge is not None
    assert start_edge.min_start_delay_us == (30.0 - 25.0) * 1e6
    # No dangling edges to the dropped predecessors.
    assert all(e.source in surviving or e.source == "START" for e in edges)
    assert all(e.target in surviving for e in edges)


def test_chop_keeps_surviving_inter_turn_edge_delay() -> None:
    """Two surviving turns keep their recorded inter-turn ``delay_after_predecessor_us``.

    t* = 5 s drops only turn 0 (offset 0). Turns 1,2,3 survive. turn1 (offset 10)
    re-roots from START at offset 5 s. The surviving turn1->turn2 and turn2->turn3
    edges keep their recorded 9 s ``delay_after_predecessor_us``.
    """
    graph, _ = _build(_four_turn_trace())
    ids = _ordered_node_ids(graph)
    t_star_us = int(5.0 * 1e6)

    chopped = chop_trie_at_tstar(graph, t_star_us)

    surviving = _llm_nodes(chopped)
    assert set(surviving) == {ids[1], ids[2], ids[3]}

    edges = _edges(chopped)
    # turn1 re-rooted from START at t*-relative offset (10 - 5 = 5 s).
    start_edge = _edge(edges, "START", ids[1])
    assert start_edge is not None
    assert start_edge.min_start_delay_us == (10.0 - 5.0) * 1e6
    # turn1 is no longer rooted from its dropped predecessor (turn0).
    assert _edge(edges, ids[0], ids[1]) is None

    # Surviving inter-turn edges keep their recorded 9 s gap.
    e12 = _edge(edges, ids[1], ids[2])
    e23 = _edge(edges, ids[2], ids[3])
    assert e12 is not None and e12.delay_after_predecessor_us == 9.0 * 1e6
    assert e23 is not None and e23.delay_after_predecessor_us == 9.0 * 1e6
    # Surviving inter-turn edges are NOT re-rooted from START.
    assert _edge(edges, "START", ids[2]) is None
    assert _edge(edges, "START", ids[3]) is None


def test_chop_leaves_prompt_segment_ids_unchanged() -> None:
    """A surviving re-rooted node keeps its FULL pre-t* prompt path verbatim."""
    graph, _ = _build(_four_turn_trace())
    ids = _ordered_node_ids(graph)
    before = _llm_nodes(graph)[ids[3]].metadata["trie"]["prompt_segment_ids"]

    chopped = chop_trie_at_tstar(graph, int(25.0 * 1e6))

    after = _llm_nodes(chopped)[ids[3]].metadata["trie"]["prompt_segment_ids"]
    assert after == before
    assert len(after) > 1  # full lineage prefix, not truncated to the resume turn


def test_chop_tstar_zero_returns_graph_unchanged() -> None:
    """``t* <= 0`` is full replay: the graph is returned identically."""
    graph, _ = _build(_four_turn_trace())

    assert chop_trie_at_tstar(graph, 0) is graph
    assert chop_trie_at_tstar(graph, -1) is graph


def test_chop_drops_dropped_predecessor_input() -> None:
    """A 2-pred node losing ONE pred to the chop keeps only the surviving input.

    t* = 1.1 s drops p0 (offset 0) and s1_inner (offset 1.0) but keeps s2_inner
    (1.2 s) and the resume turn p1 (4 s). p1's ``inputs`` must drop the dropped
    ``s1_inner_out`` requirement and keep only ``s2_inner_out`` -- the dropped
    channel is never written post-chop, so a stale requirement would DEADLOCK
    ``await_inputs``. p1 still has a surviving predecessor (s2_inner), so it is
    NOT re-rooted from START.
    """
    graph, _ = _build(_join_trace())
    nodes = _llm_nodes(graph)
    by_offset = {n.arrival_offset_us: nid for nid, n in nodes.items()}
    s1_inner = by_offset[int(1.0 * 1e6)]
    s2_inner = by_offset[int(1.2 * 1e6)]
    p1 = by_offset[int(4.0 * 1e6)]
    # Precondition: p1 fans in on its interval-order frontier -- BOTH subagent
    # last turns (two predecessors) -- before the chop. The content-parent p0 is
    # transitively covered and is NOT a direct predecessor under interval-order.
    assert set(nodes[p1].inputs) == {
        ChannelRequirement(channel=f"{s1_inner}_out", count=1),
        ChannelRequirement(channel=f"{s2_inner}_out", count=1),
    }

    chopped = chop_trie_at_tstar(graph, int(1.1 * 1e6))

    survivors = _llm_nodes(chopped)
    assert p1 in survivors and s2_inner in survivors
    # p1 keeps ONLY the surviving predecessor's input requirement.
    assert survivors[p1].inputs == [
        ChannelRequirement(channel=f"{s2_inner}_out", count=1)
    ]
    # The surviving inter-turn edge s2_inner -> p1 remains; p1 is not re-rooted.
    edges = _edges(chopped)
    assert _edge(edges, s2_inner, p1) is not None
    assert _edge(edges, "START", p1) is None


def test_chop_dropping_all_predecessors_reroots_from_start_with_no_inputs() -> None:
    """A node losing ALL preds to the chop re-roots from START with empty inputs.

    t* = 2.0 s drops p0, s1_inner, s2_inner (offsets 0, 1.0, 1.2) but keeps p1
    (offset 4 s). p1 lost BOTH predecessors, so it re-roots from START and its
    ``inputs`` collapse to empty -- otherwise ``await_inputs`` would block on
    channels no surviving node ever writes.
    """
    graph, _ = _build(_join_trace())
    nodes = _llm_nodes(graph)
    by_offset = {n.arrival_offset_us: nid for nid, n in nodes.items()}
    p1 = by_offset[int(4.0 * 1e6)]

    chopped = chop_trie_at_tstar(graph, int(2.0 * 1e6))

    survivors = _llm_nodes(chopped)
    assert set(survivors) == {p1}
    # No predecessor survived: inputs collapse to empty.
    assert survivors[p1].inputs == []
    # Re-rooted from START at its t*-relative offset (4 - 2 = 2 s).
    edges = _edges(chopped)
    start_edge = _edge(edges, "START", p1)
    assert start_edge is not None
    assert start_edge.min_start_delay_us == (4.0 - 2.0) * 1e6
