# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``TrieNode.causal_parent_id`` stamping by ``dynamo_trie_nodes``: prev-in-chain for turns k>0, latest parent-session turn started at or before a subagent's first turn, and None for chain roots or when no earlier parent turn exists."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import _Chain, _Turn
from aiperf.dataset.graph.adapters.dynamo.trace_reader import AgentTraceRecord
from aiperf.dataset.graph.adapters.dynamo.trie_lowering import dynamo_trie_nodes


def _rec_obj(
    ts: int,
    sid: str,
    itok: int,
    otok: int,
    received: int | None = None,
    total: int | None = None,
) -> AgentTraceRecord:
    """One current-schema ``dynamo.request.trace.v1`` ``request_end`` record object."""
    req: dict[str, Any] = {
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


def _chain(
    sid: str, turns: list[AgentTraceRecord], parent: str | None = None
) -> _Chain:
    """One ``_Chain`` of already-built records, optionally hung off a parent session."""
    return _Chain(
        sid,
        parent_session_id=parent,
        turns=[_Turn(record=r) for r in turns],
    )


def _two_turn_chain() -> dict[str, _Chain]:
    """Single root session with two turns, 1s apart."""
    return {
        "s1": _chain(
            "s1",
            [
                _rec_obj(1000, "s1", 32, 8, received=1000),
                _rec_obj(2000, "s1", 64, 8, received=2000),
            ],
        )
    }


def _child_after_second_parent_turn() -> dict[str, _Chain]:
    """Subagent starting at 5.0s under a root whose turns start at 0.0s and 4.0s."""
    return {
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


def _single_root_turn() -> dict[str, _Chain]:
    """Single root session with exactly one turn."""
    return {"s1": _chain("s1", [_rec_obj(1000, "s1", 32, 8, received=1000)])}


def _child_before_any_parent_turn() -> dict[str, _Chain]:
    """Subagent starting at 1.0s under a root whose only turn starts later, at 5.0s."""
    return {
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


@pytest.mark.parametrize(
    "make_chains,node_id,expected_causal_parent",
    [
        param(_two_turn_chain, "s1:1", "s1:0", id="turn_k_gt_0_stamps_prev_in_chain"),
        param(
            _child_after_second_parent_turn,
            "child:0",
            "root:1",
            id="child_first_turn_stamps_latest_parent_turn_at_or_before_start",
        ),
        param(_single_root_turn, "s1:0", None, id="root_first_turn_has_no_parent"),
        param(
            _child_before_any_parent_turn,
            "child:0",
            None,
            id="child_first_turn_before_any_parent_turn_has_no_parent",
        ),
    ],
)  # fmt: skip
def test_causal_parent_stamping(
    make_chains: Callable[[], dict[str, _Chain]],
    node_id: str,
    expected_causal_parent: str | None,
) -> None:
    """``dynamo_trie_nodes`` stamps each node's causal parent from chain order and parent-session start times."""
    nodes, _bs, _tags = dynamo_trie_nodes(make_chains())
    by_id = {n.node_id: n for n in nodes}
    assert by_id[node_id].causal_parent_id == expected_causal_parent
