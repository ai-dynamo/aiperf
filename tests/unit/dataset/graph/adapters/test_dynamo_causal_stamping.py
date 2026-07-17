# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for dynamo causal-parent stamping in the shared-trie lowering.

Covers ``TrieNode.causal_parent_id`` as filled by ``dynamo_trie_nodes``:
prev-in-chain for turns k>0 (0-based), the latest parent-session turn started
at or before a subagent's first turn, and ``None`` for chain roots / no earlier
parent turn.
"""

from __future__ import annotations

from aiperf.dataset.graph.adapters.dynamo.trace import _Chain, _Turn
from aiperf.dataset.graph.adapters.dynamo.trace_reader import AgentTraceRecord
from aiperf.dataset.graph.adapters.dynamo.trie_lowering import dynamo_trie_nodes


def _rec_obj(ts, sid, itok, otok, received=None, total=None):
    req = {
        "request_id": f"r{ts}",
        "model": "m",
        "input_tokens": itok,
        "output_tokens": otok,
        "cached_tokens": 0,
    }
    if received is not None:
        req["request_received_ms"] = received
    if total is not None:
        req["total_time_ms"] = total
    return AgentTraceRecord.model_validate(
        {
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": ts,
            "event_source": "dynamo",
            "agent_context": {"session_id": sid},
            "request": req,
        }
    )


def _chain(sid, turns, parent=None):
    return _Chain(
        sid,
        parent_session_id=parent,
        turns=[_Turn(record=r) for r in turns],
    )


def _by_id(nodes):
    return {n.node_id: n for n in nodes}


def test_second_turn_stamps_previous_turn_in_chain():
    """Turn k>0 (0-based) causal parent is the prior turn in the same chain."""
    chains = {
        "s1": _chain(
            "s1",
            [
                _rec_obj(1000, "s1", 32, 8, received=1000),
                _rec_obj(2000, "s1", 64, 8, received=2000),
            ],
        )
    }
    nodes, _bs, _tags = dynamo_trie_nodes(chains)
    by_id = _by_id(nodes)
    assert by_id["s1:1"].causal_parent_id == "s1:0"


def test_child_first_turn_stamps_latest_parent_turn_at_or_before_start():
    """A subagent's first turn (start 5.0) picks the latest parent-session turn
    whose start (0.0, 4.0) is <= its own start -- the 4.0 turn."""
    chains = {
        "root": _chain(
            "root",
            [
                _rec_obj(0, "root", 32, 8, received=0),
                _rec_obj(4000, "root", 64, 8, received=4000),
            ],
        ),
        "child": _chain(
            "child",
            [_rec_obj(5000, "child", 16, 4, received=5000)],
            parent="root",
        ),
    }
    nodes, _bs, _tags = dynamo_trie_nodes(chains)
    by_id = _by_id(nodes)
    assert by_id["child:0"].causal_parent_id == "root:1"


def test_root_first_turn_has_no_causal_parent():
    """Turn k==0 of a root chain (no parent_session_id) is None."""
    chains = {"s1": _chain("s1", [_rec_obj(1000, "s1", 32, 8, received=1000)])}
    nodes, _bs, _tags = dynamo_trie_nodes(chains)
    assert _by_id(nodes)["s1:0"].causal_parent_id is None


def test_child_first_turn_before_any_parent_turn_has_no_causal_parent():
    """A subagent's first turn whose parent session has NO turn started at or
    before it resolves to None."""
    chains = {
        "root": _chain(
            "root",
            [_rec_obj(5000, "root", 32, 8, received=5000)],
        ),
        "child": _chain(
            "child",
            [_rec_obj(1000, "child", 16, 4, received=1000)],
            parent="root",
        ),
    }
    nodes, _bs, _tags = dynamo_trie_nodes(chains)
    assert _by_id(nodes)["child:0"].causal_parent_id is None
