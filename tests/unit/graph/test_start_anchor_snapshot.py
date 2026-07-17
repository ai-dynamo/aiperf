# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start-anchored edges survive the snapshot + pacing consumers.

Two consumers read ``StaticEdge.delay_after_predecessor_start_us`` (Task 3
stamping) besides the executor (Task 5):

1. ``chop_trie_at_tstar`` (``timing.snapshot_chop``, the segment-trie t*
   frontier chop) keys on which SOURCES survive, not on the delay KIND -- so it
   drops a start-anchored parent (re-rooting the child from START) and keeps a
   surviving start-anchored edge VERBATIM without adding a competing synthetic
   START anchor (no double-counted ``min_start_delay_us``). The ``test_chop_*``
   cases LOCK that behavior through the real chop path.
2. ``GraphIRReplayStrategy._max_inter_turn_gap_seconds`` scans every edge delay
   for the idle-gap advisory; a start-anchored delay is an inter-turn gap too.

Uses the overlap fixture from ``test_start_anchor_runtime.py`` (Task 5), copied
here so the module is self-contained. Geometry: ``START->start_anchor:0``;
``start_anchor:0->a1:0`` start-delay 2.5e6; ``start_anchor:0->start_anchor:1``
start-delay 5.0e6; ``start_anchor:0->start_anchor:2`` end-delay 1.0e6
+ ``start_anchor:1->start_anchor:2`` end-delay 0.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
from aiperf.dataset.graph.adapters.weka.trie_build import (
    ReconCallbacks,
    build_trie_graph,
)
from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.timing.snapshot_chop import chop_trie_at_tstar
from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

_BLOCK_SIZE = 64


def _stub_decode_block_tokens(hash_ids: list[int]) -> list[int]:
    out: list[int] = []
    for h in hash_ids:
        out.extend(range(h * 100, h * 100 + _BLOCK_SIZE))
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

# P: t=0 spawner; C: subagent first at t=2.5 (P in flight); Q: chain-overlap at
# t=5.0 (P in flight); R: t=9.0 (after P ends, end-anchored). Builds
# ``start_anchor:0 -> a1:0`` / ``start_anchor:0 -> start_anchor:1`` start-anchored
# edges + a ``start_anchor:1 -> start_anchor:2`` end-anchored survivor edge.
_OVERLAP_TRACE = {
    "id": "start_anchor", "models": ["M"], "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "n", "model": "M", "in": 128, "out": 64,
         "hash_ids": [1, 2], "api_time": 8.0, "stop": "tool_use"},
        {"t": 2.0, "type": "subagent", "agent_id": "a1",
         "subagent_type": "Explore", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 2.5, "type": "n", "model": "M", "in": 128, "out": 16,
              "hash_ids": [50, 51], "api_time": 1.0},
         ]},
        {"t": 5.0, "type": "n", "model": "M", "in": 192, "out": 32,
         "hash_ids": [1, 2, 3], "api_time": 1.0},
        {"t": 9.0, "type": "n", "model": "M", "in": 256, "out": 32,
         "hash_ids": [1, 2, 3, 4], "api_time": 0.5},
    ],
}  # fmt: skip


# --- chop: LOCK the already-correct behavior (SOURCE survival, not delay kind) -


def test_chop_drops_parent_rewrites_child_to_absolute_start():
    """A dropped start-anchored parent re-roots its child from START.

    t* = 3.0s: P (arrival 0) and C (2.5) are warmup-dropped; Q (5.0) survives.
    Q's only predecessor (P) is chopped, so Q re-roots from START at its
    t*-relative absolute offset -- the synthetic START edge carries NO
    start-anchored delay.
    """
    parsed, _pool = build_trie_graph(
        WekaTrace.model_validate(_OVERLAP_TRACE), callbacks=_STUB_CALLBACKS
    )
    chopped = chop_trie_at_tstar(parsed, t_star_us=3_000_000)
    (q_edge,) = [e for e in chopped.graph.edges if e.target == "start_anchor:1"]
    assert q_edge.source == "START"
    assert q_edge.min_start_delay_us == pytest.approx(2_000_000)
    assert q_edge.delay_after_predecessor_start_us is None


def test_chop_t_star_zero_short_circuits_graph_unchanged():
    """t* <= 0 returns the parsed graph object UNCHANGED (documented contract)."""
    parsed, _pool = build_trie_graph(
        WekaTrace.model_validate(_OVERLAP_TRACE), callbacks=_STUB_CALLBACKS
    )
    assert chop_trie_at_tstar(parsed, t_star_us=0) is parsed


def _shifted_requests(
    requests: list[dict[str, Any]], dt: float
) -> list[dict[str, Any]]:
    """Shift every recorded ``t`` (including nested subagent bodies) by ``dt``."""
    shifted: list[dict[str, Any]] = []
    for req in requests:
        req = dict(req)
        req["t"] = req["t"] + dt
        if "requests" in req:
            req["requests"] = _shifted_requests(req["requests"], dt)
        shifted.append(req)
    return shifted


def test_chop_keeps_surviving_start_anchored_edge_verbatim():
    """A surviving start-anchored edge is kept verbatim THROUGH the chop path.

    ``t* = 0`` short-circuits ``chop_trie_at_tstar`` entirely (graph returned
    unchanged), which would leave the keep-verbatim branch unexecuted. Shifting
    the trace +2s and chopping at t*=1s keeps every node while forcing the real
    edge-recompute path to run: the start-anchored survivor edge must come out
    unchanged, with no competing synthetic START anchor on its target (no
    double-counted ``min_start_delay_us``).
    """
    trace = dict(
        _OVERLAP_TRACE, requests=_shifted_requests(_OVERLAP_TRACE["requests"], 2.0)
    )
    parsed, _pool = build_trie_graph(
        WekaTrace.model_validate(trace), callbacks=_STUB_CALLBACKS
    )
    chopped = chop_trie_at_tstar(parsed, t_star_us=1_000_000)
    assert chopped is not parsed, "t*>0 must run the real chop path"
    assert set(chopped.graph.nodes) == set(parsed.graph.nodes), "nothing dropped"

    (c_edge,) = [e for e in chopped.graph.edges if e.target == "a1:0"]
    assert c_edge.source == "start_anchor:0"
    assert c_edge.delay_after_predecessor_start_us == pytest.approx(2.5e6)
    assert c_edge.min_start_delay_us is None, (
        "surviving start-anchored edge must not gain a START-style offset"
    )


def test_max_inter_turn_gap_includes_start_anchored_delays():
    """A corpus whose only delay is start-anchored reports it as the max gap."""
    nodes = {
        "A": LlmNode(prompt=["@a"], output="a"),
        "B": LlmNode(prompt=["@b"], output="b"),
    }
    edges = [
        StaticEdge(source="START", target="A"),
        StaticEdge(source="A", target="B", delay_after_predecessor_start_us=7_000_000),
    ]
    parsed = ParsedGraph(
        graph=GraphRecord(nodes=nodes, edges=edges, state={}),
        traces=[TraceRecord(id="t")],
    )
    stub = SimpleNamespace(_parsed=parsed)

    gap_s = GraphIRReplayStrategy._max_inter_turn_gap_seconds(stub)

    assert gap_s == pytest.approx(7.0)
