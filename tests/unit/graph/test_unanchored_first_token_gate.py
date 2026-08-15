# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A first-token-anchored edge that carries no start anchor is rejected at graph load.

Pins the invariant the executor's first-token wait silently depends on.
``TraceExecutor._apply_firing_delay`` blocks on
``ctx.first_token_event(edge.source).wait()`` for every incoming edge carrying
``delay_after_predecessor_first_token_us``. Nothing in that wait bounds itself:
``EXECUTOR_WATCHDOG_TIMEOUT`` is unset by default, so a source that never runs
parks the target forever.

Today the wait is safe, but only as an emergent property of three files
agreeing:

1. ``interval_order.apply_start_anchors`` -- the sole producer -- always sets
   ``delay_after_predecessor_start_us`` on the SAME edge, and replaces the
   target's whole in-edge set with that one edge.
2. ``Scheduler.__init__`` classifies any edge carrying
   ``delay_after_predecessor_start_us`` as start-anchored.
3. ``_reject_mixed_anchor_fan_in`` then guarantees a start-anchored in-edge is
   its target's ONLY in-edge -- so the target is scheduled solely by
   ``start_anchored_successors(source)`` at the source's DISPATCH, meaning the
   source is already running and its ``finally`` will set the latch.

Break link 1 -- a future lowering emitting a first-token delay WITHOUT a start
delay -- and links 2 and 3 stop applying: the target becomes reachable from an
unrelated predecessor while its first-token source never runs, and the wait
deadlocks with no diagnostic. These tests pin the invariant locally so that
regression fails at graph load instead of hanging a benchmark.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.dataset.graph.models import GraphRecord, LlmNode, StaticEdge
from aiperf.dataset.graph.segment_trie.interval_order import apply_start_anchors
from aiperf.graph.scheduler import Scheduler


def _llm(output: str) -> LlmNode:
    return LlmNode(prompt=[f"@{output}"], output=output)


def _graph(edges: list[StaticEdge]) -> GraphRecord:
    return GraphRecord(
        nodes={"a": _llm("a"), "b": _llm("b"), "d": _llm("d")},
        edges=[StaticEdge(source="START", target="a"), *edges],
        state={},
    )


def test_scheduler_rejects_first_token_edge_without_start_anchor() -> None:
    """The regression shape: a first-token delay with no start delay on the same edge."""
    graph = _graph(
        [
            StaticEdge(
                source="a",
                target="d",
                delay_after_predecessor_first_token_us=1_000.0,
            )
        ]
    )

    with pytest.raises(NotImplementedError) as exc_info:
        Scheduler(graph)

    msg = str(exc_info.value)
    # Validator-gate convention: "<loc>: <reason>" naming the offending node.
    assert msg.startswith("node 'd': ")
    assert "'a' -> 'd'" in msg
    assert "delay_after_predecessor_first_token_us" in msg
    assert "delay_after_predecessor_start_us" in msg


def test_rejection_survives_a_completion_anchor_on_the_same_edge() -> None:
    """A completion anchor does NOT substitute for the start anchor.

    ``delay_after_predecessor_us`` leaves the edge classified as a completion
    edge, so the target stays reachable from other predecessors and the
    sole-in-edge guarantee never applies -- exactly the unsafe shape.
    """
    graph = _graph(
        [
            StaticEdge(
                source="a",
                target="d",
                delay_after_predecessor_us=500.0,
                delay_after_predecessor_first_token_us=1_000.0,
            )
        ]
    )

    with pytest.raises(NotImplementedError, match="node 'd': "):
        Scheduler(graph)


@pytest.mark.parametrize(
    "edge",
    [
        param(
            StaticEdge(
                source="a",
                target="d",
                delay_after_predecessor_start_us=2_000.0,
                delay_after_predecessor_first_token_us=1_000.0,
            ),
            id="first-token-with-start-anchor",
        ),
        param(
            StaticEdge(
                source="a", target="d", delay_after_predecessor_start_us=2_000.0
            ),
            id="start-anchor-alone",
        ),
        param(
            StaticEdge(source="a", target="d", delay_after_predecessor_us=2_000.0),
            id="completion-anchor-alone",
        ),
        param(StaticEdge(source="a", target="d"), id="no-anchor"),
    ],
)  # fmt: skip
def test_scheduler_accepts_every_well_formed_anchor_shape(edge: StaticEdge) -> None:
    """The gate must reject ONLY the unanchored first-token shape."""
    Scheduler(_graph([edge]))


def test_apply_start_anchors_never_emits_the_rejected_shape() -> None:
    """The sole producer upholds the invariant the gate now pins.

    Guards link 1 directly: a streaming parent whose child was recorded at/after
    the recorded first token is the one case that mints
    ``delay_after_predecessor_first_token_us``, and it must always co-emit the
    start anchor.
    """
    from aiperf.dataset.graph.segment_trie.trie_content import TrieNode, TrieRequest

    def _req(t: float, api_time: float, ttft: float | None = None) -> TrieRequest:
        return TrieRequest(
            hash_ids=[],
            input_length=1,
            output_length=1,
            t=t,
            api_time=api_time,
            streaming=ttft is not None,
            ttft=ttft,
        )

    # Streaming parent occupying [0, 10) with a recorded first token at 1s.
    parent = TrieNode(node_id="p", request=_req(0.0, 10.0, ttft=1.0), order=0)
    # Child starts at t=2s -- inside the parent's interval AND after its first
    # token, the one shape that mints delay_after_predecessor_first_token_us.
    child = TrieNode(node_id="c", request=_req(2.0, 3.0), order=1, causal_parent_id="p")
    for node in (parent, child):
        node.warped_start = node.request.t

    edges_by_node: dict[str, list[StaticEdge]] = {}
    apply_start_anchors([parent, child], edges_by_node)

    emitted = edges_by_node["c"]
    assert len(emitted) == 1, "a start anchor replaces the WHOLE in-edge set"
    edge = emitted[0]
    assert edge.delay_after_predecessor_first_token_us is not None, (
        "post-TTFT child should be first-token anchored; fixture no longer "
        "exercises the invariant"
    )
    assert edge.delay_after_predecessor_start_us is not None, (
        "a first-token anchor MUST co-emit a start anchor -- without it the "
        "Scheduler's sole-in-edge guarantee does not apply and the executor's "
        "first-token wait can deadlock"
    )
