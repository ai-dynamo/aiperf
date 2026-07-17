# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-only ``ParsedGraph`` construction from a ``WekaTrace`` (Task 3).

:func:`build_trie_graph` walks a Weka trace (top-level + nested subagent
requests) in recorded time order and emits a trivial ``ParsedGraph``: one
``LlmNode`` per recorded ``n``/``s`` request and plain ``StaticEdge``
"waits-for" dependency edges. NO reducers / channels / subgraphs / spawn /
await nodes; NO chain-detection / ``::fa``-``::aux`` classification.

The edge rule (the heart) is interval order plus a start-anchor carve-out:
  * ``A -> R`` iff A FINISHED before R STARTED (raw ``A.t + A.api_time <= R.t``)
    AND ``rank(A) < rank(R)``, after async exclusion and a frontier
    (transitive-reduction) filter. The latest-ending frontier predecessor
    carries the warped end-to-start ``delay_after_predecessor_us``; every other
    frontier predecessor is an AND-fan-in wait (delay 0).
  * A request with NO finished-before cause roots at ``StaticEdge(START, R)``
    with ``min_start_delay_us = R.t * 1e6``.
  * Start-anchor carve-out: when R's recorded causal parent (spawner / chain-prev)
    was still IN FLIGHT at R's start, R's incoming edges collapse to ONE
    start-anchored ``StaticEdge(parent -> R, delay_after_predecessor_start_us =
    warped start-to-start gap)`` -- 0 for coincident-start siblings, which
    therefore dispatch together (see
    ``test_coincident_fanout_second_branch_start_anchors_to_chain_prev``).
  * Concurrency is otherwise emergent: two requests sharing a cause with no edge
    between them stay edge-free; the only inter-sibling edge the builder adds is
    the start-anchor above.

These tests drive the builder with the same deterministic stub callbacks the
Task-2 segment-emission test uses, so no tokenizer / corpus build is needed;
they assert dependency STRUCTURE (edges, fan-in, fan-out), not content bytes.
"""

from __future__ import annotations

import pytest

from aiperf.dataset.graph.adapters.weka.trace import EmptyWekaTraceError
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

BLOCK_SIZE = 64


def _stub_decode_block_tokens(hash_ids: list[int]) -> list[int]:
    """Return ``BLOCK_SIZE`` distinct token IDs per hash id (agentx stub parity)."""
    out: list[int] = []
    for h in hash_ids:
        out.extend(range(h * 100, h * 100 + BLOCK_SIZE))
    return out


def _stub_partial_tail_tokens(n_tokens: int, seed: str) -> list[int]:
    """Return ``n_tokens`` deterministic token IDs keyed on ``seed`` (stub)."""
    base = sum(ord(c) for c in seed) * 1000
    return list(range(base, base + n_tokens))


def _stub_decode_tokens_to_text(tokens: list[int]) -> str:
    """Decode a token list to text by joining IDs (collision-free stub)."""
    return "|".join(str(t) for t in tokens)


_STUB_CALLBACKS = ReconCallbacks(
    decode_block_tokens=_stub_decode_block_tokens,
    sample_partial_tail_tokens=_stub_partial_tail_tokens,
    decode_tokens_to_text=_stub_decode_tokens_to_text,
)


def _build(trace: WekaTrace) -> tuple[ParsedGraph, SegmentPool]:
    """Run the builder with the deterministic stub reconstructor callbacks."""
    return build_trie_graph(trace, callbacks=_STUB_CALLBACKS)


def _llm_nodes(graph: ParsedGraph) -> dict[str, LlmNode]:
    """All ``LlmNode``s in the single top-level graph keyed by node id."""
    return {nid: n for nid, n in graph.graph.nodes.items() if isinstance(n, LlmNode)}


def _edges(graph: ParsedGraph) -> list[StaticEdge]:
    """All ``StaticEdge``s in the single top-level graph."""
    return [e for e in graph.graph.edges if isinstance(e, StaticEdge)]


def _edge(edges: list[StaticEdge], source: str, target: str) -> StaticEdge | None:
    """Find the (first) edge ``source -> target`` if present."""
    for e in edges:
        if e.source == source and e.target == target:
            return e
    return None


def _n(t: float, hashes: list[int], in_len: int, out: int, api_time: float,
       model: str = "M", type_: str = "n") -> dict:  # fmt: skip
    """One normal/streaming request dict in the Weka wire shape."""
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
    """One subagent marker dict with nested inner requests."""
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


def _response_segment(pool, node):
    """The node's assistant response segment: the assistant pool entry chained
    onto the node's prompt tip (content-addressed, so a successor's identical
    history message dedups onto the same entry)."""
    tip = node.metadata["trie"]["prompt_segment_ids"][-1]
    return next(
        s for s in pool.by_id.values() if s.role == "assistant" and s.parent_id == tip
    )


def _trace(requests: list[dict]) -> WekaTrace:
    """Schema-validate a minimal local-scope Weka trace."""
    return WekaTrace.model_validate(
        {
            "id": "trace_0",
            "models": ["M"],
            "block_size": BLOCK_SIZE,
            "hash_id_scope": "local",
            "requests": requests,
        }
    )


def test_sequential_continuation_single_predecessor() -> None:
    """Two turns where turn1 starts after turn0 finishes -> edge turn0->turn1.

    turn0: t=0, api_time=2 -> completes at 2.0. turn1: t=3 (>= 2.0) shares
    turn0's hash prefix (content-parent). The completed-before test passes, so
    a single ``StaticEdge`` turn0->turn1 is emitted with
    ``delay_after_predecessor_us = (3 - 2) * 1e6``. turn0 itself roots at START.
    """
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=2.0),
            _n(t=3.0, hashes=[1, 2, 3, 4], in_len=256, out=64, api_time=1.0),
        ]
    )
    graph, pool = _build(trace)

    nodes = _llm_nodes(graph)
    assert len(nodes) == 2
    ids = list(nodes.keys())
    n0, n1 = ids[0], ids[1]

    edges = _edges(graph)
    # turn0 roots at START; turn1 waits for turn0.
    assert _edge(edges, "START", n0) is not None
    e = _edge(edges, n0, n1)
    assert e is not None
    assert e.delay_after_predecessor_us == (3.0 - 2.0) * 1e6
    # No spurious edge back from turn1 or START->turn1.
    assert _edge(edges, "START", n1) is None
    assert _edge(edges, n1, n0) is None

    # Dispatch overrides + arrival offset on the successor.
    assert nodes[n1].arrival_offset_us == int(3.0 * 1e6)
    assert nodes[n1].max_tokens == 64
    assert nodes[n1].model == "M"
    # trie metadata carries the prompt path; the response segment is the
    # assistant pool entry chained onto the prompt tip.
    trie = nodes[n1].metadata["trie"]
    assert set(trie) == {"prompt_segment_ids"}
    assert trie["prompt_segment_ids"]
    assert _response_segment(pool, nodes[n1]).role == "assistant"


def test_coincident_fanout_second_branch_start_anchors_to_chain_prev() -> None:
    """B and C both branch off A at the SAME instant -> C start-anchors to B.

    A: t=0, api_time=1 -> done at 1.0. B (t=2) and C (t=2) both extend A's hash
    prefix and run concurrently. B completed-before-waits on A (end-anchored
    ``A -> B``). C's positional chain-prev is B, and B is still in flight at C's
    coincident start (2 <= 2 < 7), so the start-anchor post-pass collapses C's
    incoming edges to ONE start-anchored ``B -> C`` edge with a ZERO start-to-start
    delay. C therefore dispatches WITH B (same warped instant t=2) -- the recorded
    concurrency is preserved, just expressed as a start-anchored edge instead of a
    second end-anchored fan-out from A. No backward ``C -> B`` edge; neither
    sibling re-roots at START.
    """
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=1.0),
            _n(t=2.0, hashes=[1, 2, 3], in_len=192, out=32, api_time=5.0),
            _n(t=2.0, hashes=[1, 2, 4], in_len=192, out=32, api_time=5.0),
        ]
    )
    graph, _ = _build(trace)

    nodes = list(_llm_nodes(graph).keys())
    a, b, c = nodes[0], nodes[1], nodes[2]

    edges = _edges(graph)
    assert _edge(edges, "START", a) is not None
    # B end-anchored on its completed-before cause A (A finished at 1.0 < 2.0).
    ab = _edge(edges, a, b)
    assert ab is not None and ab.delay_after_predecessor_start_us is None
    # C's incoming collapses to a single start-anchored edge from its chain-prev B.
    assert _edge(edges, a, c) is None
    c_incoming = [e for e in edges if e.target == c]
    assert len(c_incoming) == 1
    (bc,) = c_incoming
    assert bc.source == b
    assert bc.delay_after_predecessor_start_us == 0.0
    # No backward inter-sibling edge; neither sibling re-roots at START.
    assert _edge(edges, c, b) is None
    assert _edge(edges, "START", b) is None
    assert _edge(edges, "START", c) is None


def test_blocking_join_and_fan_in() -> None:
    """Parent spawns subagents s1,s2; resume turn waits for BOTH last-turns.

    The parent's first turn (p0) precedes two completed subagent markers. The
    subagents' fresh-context first turns get a structural-spawner edge from p0.
    The parent's resume turn (p1) starts after both subagents finish, so it
    AND-fans-in: an edge from s1's last turn AND from s2's last turn.
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
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=0.5),
            s1,
            s2,
            # Resume turn extends the parent prefix and starts after both
            # subagents complete (s1 done ~2.5, s2 done ~2.7).
            _n(t=4.0, hashes=[1, 2, 3], in_len=192, out=32, api_time=1.0),
        ]
    )
    graph, _ = _build(trace)

    nodes = _llm_nodes(graph)
    # 4 LlmNodes: p0, s1-inner, s2-inner, p1 (resume).
    assert len(nodes) == 4

    edges = _edges(graph)
    # Identify nodes by their hash prefix via the trie prompt path is overkill;
    # instead find by dispatch arrival offset / structural position.
    by_offset = {n.arrival_offset_us: nid for nid, n in nodes.items()}
    p0 = by_offset[int(0.0 * 1e6)]
    s1_inner = by_offset[int(1.0 * 1e6)]
    s2_inner = by_offset[int(1.2 * 1e6)]
    p1 = by_offset[int(4.0 * 1e6)]

    # Subagent first turns derive from the recorded spawner (p0).
    assert _edge(edges, p0, s1_inner) is not None
    assert _edge(edges, p0, s2_inner) is not None
    # The resume turn AND-fans-in on BOTH subagent last turns.
    assert _edge(edges, s1_inner, p1) is not None
    assert _edge(edges, s2_inner, p1) is not None


def test_node_inputs_match_predecessor_edges() -> None:
    """Each node's AND-fan-in ``inputs`` mirror its non-START interval-order edges.

    A multi-predecessor join node must declare one
    ``ChannelRequirement(channel="{src}_out", count=1)`` per predecessor source
    so the executor's ``await_inputs`` gate enforces the AND-join (trie LlmNodes
    otherwise declare no inputs, so the gate is a no-op and the node early-fires
    on its FIRST completing predecessor). A single-predecessor node has exactly
    one requirement; a START-rooted node has none.

    Reuses the blocking-join topology: p0 roots at START (no inputs); the two
    subagent first turns each wait on p0 (one input); the resume turn AND-fans-in
    on the interval-order FRONTIER of its finished-before causes.

    Under interval-order timing (:func:`_build_interval_edges`) p1's predecessor
    set is the MAXIMAL finished-before frontier, NOT every completed-before cause.
    p0 (end 0.5) is transitively covered -- it finished before s1_inner started
    (0.5 <= 1.0), so ``p0 -> s1_inner -> p1`` drops p0 from p1's frontier. The
    content-parent p0 is therefore NOT a direct timing predecessor of p1; p1
    fans in on the two subagent last turns ONLY (two inputs), which the executor
    still gates as an AND-join.
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
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=0.5),
            s1,
            s2,
            _n(t=4.0, hashes=[1, 2, 3], in_len=192, out=32, api_time=1.0),
        ]
    )
    graph, _ = _build(trace)

    nodes = _llm_nodes(graph)
    by_offset = {n.arrival_offset_us: nid for nid, n in nodes.items()}
    p0 = by_offset[int(0.0 * 1e6)]
    s1_inner = by_offset[int(1.0 * 1e6)]
    s2_inner = by_offset[int(1.2 * 1e6)]
    p1 = by_offset[int(4.0 * 1e6)]

    # START-rooted node has no input requirements (fires at min_start_delay_us).
    assert nodes[p0].inputs == []

    # Single-predecessor nodes wait on exactly their one spawner (p0).
    assert nodes[s1_inner].inputs == [ChannelRequirement(channel=f"{p0}_out", count=1)]
    assert nodes[s2_inner].inputs == [ChannelRequirement(channel=f"{p0}_out", count=1)]

    # The resume turn AND-fans-in on its interval-order frontier: both subagent
    # last turns. The content-parent p0 is NOT a direct predecessor -- it
    # finished before s1_inner started, so it is transitively covered
    # (p0 -> s1_inner -> p1) and dropped from p1's frontier.
    assert set(nodes[p1].inputs) == {
        ChannelRequirement(channel=f"{s1_inner}_out", count=1),
        ChannelRequirement(channel=f"{s2_inner}_out", count=1),
    }
    # p0 is not a direct timing predecessor of p1 (frontier-dropped).
    assert ChannelRequirement(channel=f"{p0}_out", count=1) not in nodes[p1].inputs
    # Inputs match the non-START predecessor edges exactly (no extras / dups).
    edges = _edges(graph)
    pred_sources = {e.source for e in edges if e.target == p1 and e.source != "START"}
    assert pred_sources == {s1_inner, s2_inner}
    assert {req.channel for req in nodes[p1].inputs} == {
        f"{src}_out" for src in pred_sources
    }
    assert len(nodes[p1].inputs) == len(pred_sources)


def test_fresh_context_subagent_uses_recorded_spawner() -> None:
    """A subagent first turn sharing NO content prefix still derives from the marker.

    The subagent's inner turn has hash_ids disjoint from the parent's, so it has
    NO content-parent. The recorded structural spawner (the parent turn before
    the marker) supplies the dependency edge instead. p0 completes at 0.5; the
    subagent starts at 1.0, so the completed-before test passes.
    """
    sa = _subagent(
        t=1.0,
        agent_id="agent_x",
        status="completed",
        duration_ms=1000,
        requests=[_n(t=1.0, hashes=[900, 901], in_len=128, out=16, api_time=0.3)],
    )
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=0.5),
            sa,
        ]
    )
    graph, _ = _build(trace)

    nodes = _llm_nodes(graph)
    assert len(nodes) == 2
    by_offset = {n.arrival_offset_us: nid for nid, n in nodes.items()}
    p0 = by_offset[int(0.0 * 1e6)]
    inner = by_offset[int(1.0 * 1e6)]

    edges = _edges(graph)
    # No content prefix -> the only cause is the recorded spawner p0.
    e = _edge(edges, p0, inner)
    assert e is not None
    assert e.delay_after_predecessor_us == (1.0 - 0.5) * 1e6
    # The fresh-context inner turn does NOT re-root at START (spawner completed).
    assert _edge(edges, "START", inner) is None


# --- max_osl dispatch cap (W2) ----------------------------------------------


def test_max_osl_caps_top_level_dispatch_not_subagent_body() -> None:
    """``max_osl`` caps top-level chain ``max_tokens``; subagent bodies stay uncapped.

    agentx ``_cap_output`` parity: the wire cap applies to the TOP-LEVEL chain
    requests only. The synthesized response SEGMENT stays sized to the recorded
    ``out`` so successor prompt content (and ISL) is unchanged.
    """
    sa = _subagent(
        t=1.0,
        agent_id="agent_x",
        status="completed",
        duration_ms=1000,
        requests=[_n(t=1.2, hashes=[900, 901], in_len=128, out=4000, api_time=0.3)],
    )
    trace = _trace(
        [
            _n(t=0.0, hashes=[1, 2], in_len=128, out=5000, api_time=0.5),
            sa,
        ]
    )
    graph, pool = build_trie_graph(trace, callbacks=_STUB_CALLBACKS, max_osl=100)
    nodes = _llm_nodes(graph)

    top = nodes["trace_0:0"]
    inner = nodes["agent_x:0"]
    assert top.max_tokens == 100, "top-level chain capped"
    assert inner.max_tokens == 4000, "subagent body uncapped"
    # The response segment is NOT capped: successor prompt bytes stay recorded.
    top_response = _response_segment(pool, top)
    assert len(top_response.content.split("|")) == 5000


def test_max_osl_above_recorded_out_is_a_noop() -> None:
    trace = _trace([_n(t=0.0, hashes=[1, 2], in_len=128, out=64, api_time=0.5)])
    graph, _ = build_trie_graph(trace, callbacks=_STUB_CALLBACKS, max_osl=10_000)
    assert _llm_nodes(graph)["trace_0:0"].max_tokens == 64


def test_max_osl_none_leaves_recorded_out_uncapped() -> None:
    trace = _trace([_n(t=0.0, hashes=[1, 2], in_len=128, out=5000, api_time=0.5)])
    graph, _ = build_trie_graph(trace, callbacks=_STUB_CALLBACKS, max_osl=None)
    assert _llm_nodes(graph)["trace_0:0"].max_tokens == 5000


# --- empty flattened leaf set (W7) ------------------------------------------


def test_all_empty_subagent_trace_raises_empty_weka_trace_error() -> None:
    """Non-empty ``requests`` that flatten to ZERO leaves must fail the parse.

    A trace whose only entries are subagent markers with empty bodies would
    otherwise produce a schedulable 0-node graph that can never fire.
    """
    trace = _trace(
        [
            _subagent(
                t=0.5,
                agent_id="agent_async",
                status="async_launched",
                requests=[],
            )
        ]
    )
    with pytest.raises(EmptyWekaTraceError, match="zero normal/streaming leaf"):
        _build(trace)
