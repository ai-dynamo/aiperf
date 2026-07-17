# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``build_trie_graph`` must honor the per-trace idle-gap cap (warp).

The trie IR derives node arrival offsets and ``StaticEdge`` delays from the
recorded ``request.t`` / ``api_time``. A recorded multi-hour IDLE gap (dead air
between one request finishing and the next starting) would otherwise survive
verbatim into the warmup phase and park it forever. ``idle_gap_cap_seconds``
builds an ``_ActiveIdleWarp`` over the UNION of every request's active interval
in the tree and places node start times, edge ``delay_after_predecessor_us``,
and root ``min_start_delay_us`` on the same idle-compressed clock. Only true
idle stretches are cut; active processing and overlapping subagents keep their
exact temporal shape.

These tests drive the builder with the deterministic stub callbacks the other
trie-build tests use, so no tokenizer / corpus build is needed; they assert the
warped TIMING geometry, not content bytes.
"""

from __future__ import annotations

from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
from aiperf.dataset.graph.adapters.weka.trie_build import (
    ReconCallbacks,
    build_trie_graph,
)
from aiperf.dataset.graph.models import LlmNode, ParsedGraph, StaticEdge
from aiperf.dataset.graph.segment_ir.pool import SegmentPool

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


def _build(trace: WekaTrace, **kwargs: object) -> tuple[ParsedGraph, SegmentPool]:
    return build_trie_graph(trace, callbacks=_STUB_CALLBACKS, **kwargs)


def _llm_nodes(graph: ParsedGraph) -> dict[str, LlmNode]:
    return {nid: n for nid, n in graph.graph.nodes.items() if isinstance(n, LlmNode)}


def _edges(graph: ParsedGraph) -> list[StaticEdge]:
    return [e for e in graph.graph.edges if isinstance(e, StaticEdge)]


def _edge(edges: list[StaticEdge], source: str, target: str) -> StaticEdge | None:
    for e in edges:
        if e.source == source and e.target == target:
            return e
    return None


def test_gap_over_cap_is_compressed() -> None:
    """A raw IDLE gap >> cap is compressed to the cap (cut between FINISH/START).

    turn0: t=0, api_time=2 -> completes at 2.0 (raw). turn1: raw t=137124, sharing
    turn0's hash prefix. The idle warp cuts the dead air between turn0 FINISHING
    (raw end 2.0) and turn1 STARTING (raw 137124) -- 137122s of idle -- down to
    the 60s cap. So turn1's warped start is ``end(2) + 60 = 62.0`` and the
    end-to-start edge delay is the capped idle ``(62 - 2) * 1e6 = 60s`` -- NOT the
    raw 137122s, and NOT ``0 + 60`` (capping start-to-start would wrongly eat into
    turn0's own 2s of processing).
    """
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=2.0),
            _n(t=137124.0, hashes=[1, 2, 3, 4], in_len=256, out=64, api_time=1.0),
        ]
    )
    graph, _ = _build(trace, idle_gap_cap_seconds=60.0)

    nodes = _llm_nodes(graph)
    ids = list(nodes.keys())
    n0, n1 = ids[0], ids[1]

    # turn1 starts 60s (capped idle) after turn0's recorded END (2.0).
    assert nodes[n1].arrival_offset_us == int(round(62.0 * 1e6))

    edges = _edges(graph)
    e = _edge(edges, n0, n1)
    assert e is not None
    # End-to-start gap on the warped clock: warped_start(62) - warped_end(2) = 60.
    assert e.delay_after_predecessor_us == 60.0 * 1e6


def test_sub_cap_gap_passes_through() -> None:
    """A raw gap below the cap is unchanged -- the warp leaves it intact."""
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=2.0),
            _n(t=3.0, hashes=[1, 2, 3, 4], in_len=256, out=64, api_time=1.0),
        ]
    )
    graph, _ = _build(trace, idle_gap_cap_seconds=60.0)

    nodes = _llm_nodes(graph)
    ids = list(nodes.keys())
    n0, n1 = ids[0], ids[1]

    assert nodes[n1].arrival_offset_us == int(round(3.0 * 1e6))
    e = _edge(_edges(graph), n0, n1)
    assert e is not None
    assert e.delay_after_predecessor_us == (3.0 - 2.0) * 1e6


def test_cap_none_is_raw_passthrough() -> None:
    """``idle_gap_cap_seconds=None`` reproduces the pre-change RAW geometry exactly.

    Builds the same trace twice -- once with the param omitted (the historic call
    shape) and once explicitly ``None`` -- and asserts both yield the RAW edge
    delays and arrival offsets (no warp applied). Regression guard for the no-cap
    path.
    """
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=2.0),
            _n(t=137124.0, hashes=[1, 2, 3, 4], in_len=256, out=64, api_time=1.0),
        ]
    )
    graph_default, _ = _build(trace)
    graph_none, _ = _build(trace, idle_gap_cap_seconds=None)

    for graph in (graph_default, graph_none):
        nodes = _llm_nodes(graph)
        ids = list(nodes.keys())
        n0, n1 = ids[0], ids[1]
        # Raw arrival offset survives -- no compression.
        assert nodes[n1].arrival_offset_us == int(round(137124.0 * 1e6))
        e = _edge(_edges(graph), n0, n1)
        assert e is not None
        assert e.delay_after_predecessor_us == (137124.0 - 2.0) * 1e6


def test_multi_gap_cumulative_shift_matches_warp() -> None:
    """3 requests with TWO >cap IDLE gaps: cumulative shift accrues across both.

    Proves the build accumulates the idle-warp shift (each prior over-cap idle
    gap shifts later events left), not a per-edge ``min(gap, cap)``: a per-edge
    clamp would lose the first gap's shift when placing the third node. Each
    capped idle sits AFTER the preceding request's END (interval warp), so:
      turn0 [0,1]; idle 1->1000 (999>cap) -> turn1 start = end(1)+60 = 61;
      turn1 [61,62] warped; idle 1001->3000 raw (1999>cap) -> turn2 start =
      warped_end(62)+60 = 122.
    """
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=1.0),
            _n(t=1000.0, hashes=[1, 2, 3], in_len=192, out=32, api_time=1.0),
            _n(t=3000.0, hashes=[1, 2, 3, 4], in_len=256, out=32, api_time=1.0),
        ]
    )
    graph, _ = _build(trace, idle_gap_cap_seconds=60.0)

    nodes = _llm_nodes(graph)
    vals = list(nodes.values())
    # turn1: end(1) + 60s capped idle = 61.
    assert vals[1].arrival_offset_us == int(round(61.0 * 1e6))
    # turn2: warped_end(62) + 60s capped idle = 122 -- cumulative across BOTH gaps.
    assert vals[2].arrival_offset_us == int(round(122.0 * 1e6))


def _subagent(t: float, agent_id: str, requests: list[dict], status: str) -> dict:
    """A blocking/async subagent marker with nested inner leaf requests."""
    return {
        "t": t,
        "type": "subagent",
        "agent_id": agent_id,
        "subagent_type": "Explore",
        "status": status,
        "duration_ms": None,
        "requests": requests,
        "models": ["M"],
    }


def _replay_starts(graph: ParsedGraph, api: dict[str, float]) -> dict[str, float]:
    """Mini discrete-event replay: fire each node at max over incoming edges of
    (sim_end(pred) + delay) [START-roots at min_start_delay; a start-anchored
    edge fires at sim_start(pred) + start_delay], occupying its recorded
    ``api_time``. Returns each node's reconstructed start. Mirrors
    tools/weka_trie_timing_sim.py so the byte-exact invariant is unit-locked."""
    incoming: dict[str, list[StaticEdge]] = {nid: [] for nid in graph.graph.nodes}
    for e in _edges(graph):
        if e.target in incoming:
            incoming[e.target].append(e)
    nodes = _llm_nodes(graph)
    order = sorted(nodes, key=lambda nid: (nodes[nid].arrival_offset_us or 0, nid))
    sim_start: dict[str, float] = {}
    sim_end: dict[str, float] = {}
    for nid in order:
        gate = 0.0
        for e in incoming[nid]:
            if e.source == "START":
                gate = max(gate, (e.min_start_delay_us or 0.0) / 1e6)
            elif e.delay_after_predecessor_start_us is not None:
                # Start-anchored: gate off the predecessor's DISPATCH, not its end.
                gate = max(
                    gate,
                    sim_start.get(e.source, 0.0)
                    + e.delay_after_predecessor_start_us / 1e6,
                )
            elif e.source in sim_end:
                gate = max(
                    gate,
                    sim_end[e.source] + (e.delay_after_predecessor_us or 0.0) / 1e6,
                )
        sim_start[nid] = gate
        sim_end[nid] = gate + api.get(nid, 0.0)
    return sim_start


def test_concurrent_turn_overlapping_its_cause_roots_at_warped_offset() -> None:
    """A subagent launched WHILE its spawner is still running start-anchors to it.

    p0 runs [0, 5]; its subagent's first turn starts at t=2 -- before p0 finished.
    It did not WAIT for p0's completion, so it must NOT get an end-anchored
    ``p0 -> s0`` completion edge; instead the causal-parent stamping + start-anchor
    post-pass collapse its incoming edges to ONE start-anchored ``p0 -> s0`` edge
    (``delay_after_predecessor_start_us``) whose delay is the warped start-to-start
    gap (t=2). The runtime schedules s0 at p0's DISPATCH and gates it 2 s later,
    so s0 still fires at its warped arrival t=2 -- concurrency preserved, no
    START re-root.
    """
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=5.0),
            _subagent(
                t=2.0,
                agent_id="a1",
                status="completed",
                requests=[
                    _n(t=2.0, hashes=[1, 2, 3], in_len=192, out=32, api_time=3.0),
                    _n(t=6.0, hashes=[1, 2, 3, 4], in_len=256, out=32, api_time=2.0),
                ],
            ),
            _n(t=9.0, hashes=[1, 2, 9], in_len=320, out=16, api_time=1.0),
        ]
    )
    graph, _ = _build(trace, idle_gap_cap_seconds=60.0)
    edges = _edges(graph)
    # s0 = r_1_0 (first inner of the subagent at index 1). It overlapped p0 (r_0),
    # so its incoming edges collapse to a single start-anchored p0 -> r_1_0 edge;
    # it does NOT re-root at START.
    assert _edge(edges, "START", "a1:0") is None
    incoming = [e for e in edges if e.target == "a1:0"]
    assert len(incoming) == 1
    (anchor_edge,) = incoming
    assert anchor_edge.source == "trace_0:0"
    assert anchor_edge.delay_after_predecessor_start_us == 2.0 * 1e6
    assert anchor_edge.delay_after_predecessor_us is None
    assert anchor_edge.min_start_delay_us is None


def test_replay_reconstructs_recorded_timeline_byte_exact() -> None:
    """Replaying the trie edges with recorded api_time lands every node on its
    recorded (warp-free here -- no >cap idle) start: p0=0, s0=2, s1=6, resume=9.

    Locks the end-to-end timing invariant the standalone simulator proves on the
    corpus (1039/1039): concurrent overlap (s0), sequential continuation (s1),
    and a parent-resume binding the final child (resume after s1 completes).
    """
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=5.0),
            _subagent(
                t=2.0,
                agent_id="a1",
                status="completed",
                requests=[
                    _n(t=2.0, hashes=[1, 2, 3], in_len=192, out=32, api_time=3.0),
                    _n(t=6.0, hashes=[1, 2, 3, 4], in_len=256, out=32, api_time=2.0),
                ],
            ),
            _n(t=9.0, hashes=[1, 2, 9], in_len=320, out=16, api_time=1.0),
        ]
    )
    graph, _ = _build(trace, idle_gap_cap_seconds=60.0)
    api = {"trace_0:0": 5.0, "a1:0": 3.0, "a1:1": 2.0, "trace_0:1": 1.0}
    sim = _replay_starts(graph, api)
    assert sim["trace_0:0"] == 0.0
    assert sim["a1:0"] == 2.0  # concurrent: launched during p0
    assert sim["a1:1"] == 6.0  # sequential after s0 (end 5) + 1s think
    assert sim["trace_0:1"] == 9.0  # parent-resume: 1s after final child s1 (end 8)
