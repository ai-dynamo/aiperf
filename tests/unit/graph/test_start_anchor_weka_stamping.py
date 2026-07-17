# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weka ``_flatten_requests`` stamps each TrieNode's ``causal_parent_id``.

The causal parent of a recorded leaf is the previous n/s leaf in its OWN list
(chain-prev), else the nearest preceding n/s leaf in an enclosing list (the
spawner), else ``None``. That stamping is what feeds
:func:`~aiperf.dataset.graph.segment_ir.interval_order.apply_start_anchors`
(already wired inside ``build_trie_ir``): when the causal parent is still IN
FLIGHT at a node's recorded start, the node's incoming edges collapse to one
start-anchored edge (``delay_after_predecessor_start_us``). The end-to-end case
also runs the graph validator to defeat the adapter-tests-skip-validator trap.
"""

from __future__ import annotations

from collections import defaultdict

import msgspec
import pytest

from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
from aiperf.dataset.graph.adapters.weka.trie_build import (
    ReconCallbacks,
    _flatten_requests,
    build_trie_graph,
)
from aiperf.dataset.graph.models import TraceRecord
from aiperf.dataset.graph.validator import ValidationSeverity, validate

_REQ = {"type": "n", "model": "M", "in": 128, "out": 16, "hash_ids": [1, 2]}

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


def _trace(requests):
    return WekaTrace.model_validate(
        {
            "id": "t",
            "models": ["M"],
            "block_size": 64,
            "hash_id_scope": "local",
            "requests": requests,
        }
    )


def test_chain_prev_stamped():
    tr = _trace(
        [
            {**_REQ, "t": 0.0, "api_time": 1.0},
            {**_REQ, "t": 2.0, "api_time": 1.0, "hash_ids": [1, 2, 3], "in": 192},
        ]
    )
    nodes = _flatten_requests(tr.requests, root_scope="t")
    assert nodes[0].causal_parent_id is None
    assert nodes[1].causal_parent_id == "t:0"


def test_subagent_first_leaf_gets_spawner():
    tr = _trace(
        [
            {**_REQ, "t": 0.0, "api_time": 8.0, "stop": "tool_use"},
            {
                "t": 2.0,
                "type": "subagent",
                "agent_id": "a1",
                "subagent_type": "Explore",
                "status": "completed",
                "models": ["M"],
                "requests": [
                    {**_REQ, "t": 2.5, "api_time": 1.0, "hash_ids": [50, 51]},
                    {
                        **_REQ,
                        "t": 4.0,
                        "api_time": 1.0,
                        "hash_ids": [50, 51, 52],
                        "in": 192,
                    },
                ],
            },
        ]
    )
    nodes = _flatten_requests(tr.requests, root_scope="t")
    by_id = {n.node_id: n for n in nodes}
    assert by_id["a1:0"].causal_parent_id == "t:0"  # spawner
    assert by_id["a1:1"].causal_parent_id == "a1:0"  # inner chain-prev


def test_marker_first_in_list_inherits_outer_spawner():
    inner = {
        "t": 1.0,
        "type": "subagent",
        "agent_id": "a2",
        "subagent_type": "Explore",
        "status": "completed",
        "models": ["M"],
        "requests": [{**_REQ, "t": 1.2, "api_time": 0.5, "hash_ids": [60, 61]}],
    }
    tr = _trace(
        [
            {**_REQ, "t": 0.0, "api_time": 4.0, "stop": "tool_use"},
            {
                "t": 0.5,
                "type": "subagent",
                "agent_id": "a1",
                "subagent_type": "Explore",
                "status": "completed",
                "models": ["M"],
                "requests": [inner],  # nested marker is FIRST entry of its list
            },
        ]
    )
    nodes = _flatten_requests(tr.requests, root_scope="t")
    by_id = {n.node_id: n for n in nodes}
    # a2:0's list has no preceding leaf; nor does its parent list; the
    # spawner is the top-level t:0.
    assert by_id["a2:0"].causal_parent_id == "t:0"


def test_end_to_end_graph_has_start_anchored_edges_and_validates():
    tr = _trace(
        [
            {**_REQ, "t": 0.0, "api_time": 8.0, "stop": "tool_use"},
            {
                "t": 2.0,
                "type": "subagent",
                "agent_id": "a1",
                "subagent_type": "Explore",
                "status": "completed",
                "models": ["M"],
                "requests": [{**_REQ, "t": 2.5, "api_time": 1.0, "hash_ids": [50, 51]}],
            },
            {**_REQ, "t": 5.0, "api_time": 1.0, "hash_ids": [1, 2, 3], "in": 192},
        ]
    )
    parsed, _pool = build_trie_graph(tr, callbacks=_STUB_CALLBACKS)
    incoming = defaultdict(list)
    for e in parsed.graph.edges:
        incoming[e.target].append(e)
    (spawn_edge,) = incoming["a1:0"]
    assert spawn_edge.source == "t:0"
    assert spawn_edge.delay_after_predecessor_start_us == pytest.approx(2.5e6)
    (chain_edge,) = incoming["t:1"]
    assert chain_edge.source == "t:0"
    assert chain_edge.delay_after_predecessor_start_us == pytest.approx(5.0e6)
    # "No completion wait" needs BOTH halves: an empty AND-fan-in alone is
    # ambiguous (a node with no gating at all also has empty inputs). The sole
    # incoming edge is the start-anchored one (destructured above) and it must
    # carry NO completion delay, so the start anchor is provably the only gate.
    assert parsed.graph.nodes["a1:0"].inputs == []
    assert spawn_edge.delay_after_predecessor_us is None
    # validator must accept the graph (adapter-tests-skip-validator trap). Attach
    # a TraceRecord so validation runs on the same trace wrap the production
    # ingest path (``_parse_trace_dict``) applies.
    validated = msgspec.structs.replace(
        parsed, traces=[TraceRecord(id=tr.id)], segment_pool=_pool
    )
    blocking = [
        i for i in validate(validated) if i.severity is ValidationSeverity.ERROR
    ]
    assert blocking == [], blocking
