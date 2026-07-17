# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-TTFT overlaps lower to first-token-anchored edges.

``apply_start_anchors`` already collapses an overlapped node's edges to one
start-anchored ``StaticEdge`` (``delay_after_predecessor_start_us``). This
module locks the post-TTFT REFINEMENT: when the recorded child starts at/after
the streaming parent's recorded first token (``node.raw_start - parent.raw_start
>= parent.request.ttft``), the same edge ALSO carries
``delay_after_predecessor_first_token_us == delay_after_predecessor_start_us -
ttft*1e6``. A pre-TTFT child (start before the recorded first token) or a
non-streaming parent (``ttft is None``) keeps the pure dispatch anchor.

The lowering is exercised at three levels: hand-built ``TrieNode``s through
``apply_start_anchors`` (mirroring ``test_start_anchor_edges.py``), the weka
adapter end-to-end via ``build_trie_graph`` (defeating the
adapter-tests-skip-validator trap), and the ``chop_trie_at_tstar`` carrier
that must round-trip the third field None-preservingly.
"""

from __future__ import annotations

from collections import defaultdict

import msgspec
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
from aiperf.dataset.graph.segment_ir.interval_order import (
    apply_start_anchors,
    build_interval_edges,
    compute_ranks,
)
from aiperf.dataset.graph.segment_ir.trie_content import (
    TrieNode,
    TrieRequest,
)
from aiperf.dataset.graph.validator import ValidationSeverity, validate
from aiperf.timing.snapshot_chop import chop_trie_at_tstar

_BLOCK_SIZE = 64


def _node(nid, t, api, causal=None, ttft=None):
    n = TrieNode(
        node_id=nid,
        request=TrieRequest(
            hash_ids=[],
            input_length=64,
            output_length=8,
            t=t,
            api_time=api,
            ttft=ttft,
        ),
        order=0,
        causal_parent_id=causal,
    )
    n.warped_start = t
    return n


# --- hand-built TrieNode edges --------------------------------------------


def test_post_ttft_overlap_gets_both_fields():
    p = _node("p", 0.0, 8.0, ttft=2.0)  # streaming parent
    c = _node("c", 4.0, 1.0, causal="p")  # 4.0 >= 0.0 + 2.0: post-TTFT
    nodes = [p, c]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)
    apply_start_anchors(nodes, edges)
    (e,) = edges["c"]
    assert e.source == "p"
    assert e.delay_after_predecessor_start_us == pytest.approx(4.0e6)
    assert e.delay_after_predecessor_first_token_us == pytest.approx(2.0e6)


def test_pre_ttft_overlap_keeps_pure_dispatch_anchor():
    p = _node("p", 0.0, 8.0, ttft=2.0)
    c = _node("c", 1.0, 1.0, causal="p")  # 1.0 < ttft: pre-TTFT
    nodes = [p, c]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)
    apply_start_anchors(nodes, edges)
    (e,) = edges["c"]
    assert e.delay_after_predecessor_start_us == pytest.approx(1.0e6)
    assert e.delay_after_predecessor_first_token_us is None


def test_no_ttft_parent_keeps_pure_dispatch_anchor():
    p = _node("p", 0.0, 8.0, ttft=None)  # non-streaming (n-type) parent
    c = _node("c", 4.0, 1.0, causal="p")  # overlaps, but parent has no ttft
    nodes = [p, c]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)
    apply_start_anchors(nodes, edges)
    (e,) = edges["c"]
    assert e.delay_after_predecessor_start_us == pytest.approx(4.0e6)
    assert e.delay_after_predecessor_first_token_us is None


# --- weka end-to-end -------------------------------------------------------


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

# ttft_anchor:0: streaming (type "s") parent, ttft 2.0, api 8.0, stops on
# tool_use. Completed subagent whose first inner request a1:0 starts at t=4.0 --
# inside ttft_anchor:0's [0, 8) interval and at/after its recorded first token
# (t >= ttft).
_TTFT_TRACE = {
    "id": "ttft_anchor", "models": ["M"], "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "s", "model": "M", "in": 128, "out": 64,
         "hash_ids": [1, 2], "api_time": 8.0, "ttft": 2.0, "stop": "tool_use"},
        {"t": 2.0, "type": "subagent", "agent_id": "a1",
         "subagent_type": "Explore", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 4.0, "type": "n", "model": "M", "in": 128, "out": 16,
              "hash_ids": [50, 51], "api_time": 1.0},
         ]},
    ],
}  # fmt: skip


def test_weka_streaming_parent_stamps_ttft_end_to_end():
    parsed, pool = build_trie_graph(
        WekaTrace.model_validate(_TTFT_TRACE), callbacks=_STUB_CALLBACKS
    )
    incoming = defaultdict(list)
    for e in parsed.graph.edges:
        incoming[e.target].append(e)
    (edge,) = incoming["a1:0"]
    assert edge.source == "ttft_anchor:0"
    assert edge.delay_after_predecessor_start_us == pytest.approx(4.0e6)
    assert edge.delay_after_predecessor_first_token_us == pytest.approx(2.0e6)
    # Validator must accept the graph (adapter-tests-skip-validator trap): the
    # first-token anchor rides its dispatch fallback on a non-START source.
    validated = msgspec.structs.replace(
        parsed, traces=[TraceRecord(id="ttft_anchor")], segment_pool=pool
    )
    blocking = [
        i for i in validate(validated) if i.severity is ValidationSeverity.ERROR
    ]
    assert blocking == [], blocking


# --- snapshot chop carrier ---------------------------------------------------


def test_chop_round_trips_first_token_anchor_on_surviving_edge():
    """``chop_trie_at_tstar`` keeps a surviving first-token-anchored edge verbatim.

    The t* chop is the snapshot carrier that rebuilds the edge set; both anchor
    fields (dispatch fallback + first-token refinement) must round-trip
    None-preservingly through the REAL chop path (t*>0, both endpoints kept).
    """
    nodes = {
        "p": LlmNode(prompt=["hi"], output="p_out", arrival_offset_us=2_000_000),
        "c": LlmNode(prompt=["hi"], output="c_out", arrival_offset_us=6_000_000),
    }
    edges = [
        StaticEdge(source="START", target="p"),
        StaticEdge(
            source="p",
            target="c",
            delay_after_predecessor_start_us=4.0e6,
            delay_after_predecessor_first_token_us=2.0e6,
        ),
    ]
    parsed = ParsedGraph(
        graph=GraphRecord(nodes=nodes, edges=edges, state={}),
        traces=[TraceRecord(id="t")],
    )

    chopped = chop_trie_at_tstar(parsed, t_star_us=1_000_000)

    assert chopped is not parsed, "t*>0 must run the real chop path"
    (edge,) = [e for e in chopped.graph.edges if e.target == "c"]
    assert edge.source == "p"
    assert edge.delay_after_predecessor_start_us == pytest.approx(4.0e6)
    assert edge.delay_after_predecessor_first_token_us == pytest.approx(2.0e6)
    assert edge.delay_after_predecessor_us is None
