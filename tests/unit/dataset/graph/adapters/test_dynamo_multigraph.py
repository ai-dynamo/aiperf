# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo multi-graph restructure: per-tree ``TraceRecord`` + ``graph_ref``.

The dynamo adapter emits ONE ``GraphRecord`` per session-tree (mirroring the
weka multi-item path through ``merge_parsed_graphs``): each trace carries its
OWN top-level graph under ``ParsedGraph.graphs[root_session_id]`` and its
``graph_ref`` selects it. The whole-capture union is NO LONGER the emitted
shape -- ``ParsedGraph.graph`` is now the FIRST tree (back-compat), not the
union.

Two guards, both pinned at this tree:

1. **Content oracle** (byte-equivalence): the per-tree lowering CONTENT is
   identical to the pre-change single union -- only the ``ParsedGraph`` SHAPE
   changed. The pinned digests were captured from the pre-change
   ``from_dynamo_trace(fixture).graph`` (the single union). After the
   restructure, re-unioning the per-tree graphs via the retained test-only
   ``_union_graphs`` helper must reproduce those EXACT digests (nodes, edges,
   content-addressed pool keys) -- because the per-tree graphs ARE the same
   graphs the old path unioned.

2. **Scoping**: ``parsed.graphs`` is non-empty (multi-graph), each
   ``TraceRecord.graph_ref`` equals its root session id, ``resolve_trace_graph``
   returns ONLY that tree's nodes, and two distinct traces resolve to DISJOINT
   node sets.

RED on the pre-change single-union code: ``parsed.graphs`` is empty and
``graph_ref`` is ``None``, so the scoping asserts fail and the re-union oracle
sees an empty graph.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import orjson
import pytest

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.dynamo.trace import (
    _union_graphs,
    from_dynamo_trace,
)
from aiperf.dataset.graph.adapters.shared.content import CorpusContentSynthesizer
from aiperf.dataset.graph.models import LlmNode, resolve_trace_graph

_SEED = 20260706
_TOKENIZER = "builtin"

# Pinned oracle digests, captured from the PRE-CHANGE ``from_dynamo_trace().graph``
# (the single union) over the fixture below. The per-tree graphs are byte-
# identical to the graphs the old path unioned, so re-unioning them reproduces
# these EXACTLY. A drift here is a real content change, not a shape change --
# investigate, do not re-pin blindly.
# EDGES re-pinned 2026-07-07: recorded edge delays now always replay (weka
# idle-warp parity) instead of the old zeroed-delay dependency-only default;
# nodes and pool digests were unaffected, confirming content synthesis did not
# drift.
# All three re-pinned 2026-07-07 for the data-inherent node-id scheme: node
# ids became {session_id}:{k} (recorded session id verbatim, 0-based turn),
# which re-keys nodes/edges, and the trace-scoped response/tiny synthesis
# seeds shift the node-id-seeded pool entries. Hash-block prompt content is
# unchanged; the determinism guard (two independent builds agree) still holds.
_ORACLE_NODES_DIGEST = "2e825a321ea8cd7e4cb77ea5f1c86eb8"
_ORACLE_EDGES_DIGEST = "97782ee32139b01c801d0485c6976ffe"
_ORACLE_POOL_DIGEST = "f963d9998288d2d9fd1c66cd52844ad9"


@pytest.fixture(autouse=True)
def _fresh_synth_cache():
    """Isolate the process-level synthesizer cache from other tests."""
    CorpusContentSynthesizer.reset_worker_cache()
    yield
    CorpusContentSynthesizer.reset_worker_cache()


def _re(
    *,
    ts: int,
    sid: str,
    hashes: list[int],
    ilen: int,
    parent: str | None = None,
    otok: int = 8,
    bs: int = 16,
) -> dict:
    ctx: dict = {"session_id": sid, "trajectory_id": sid}
    if parent is not None:
        ctx["parent_trajectory_id"] = parent
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": ctx,
        "request": {
            "request_id": f"r-{sid}-{ts}",
            "model": "m",
            "input_tokens": ilen,
            "output_tokens": otok,
            "cached_tokens": 0,
            "ttft_ms": 10.0,
            "replay": {
                "trace_block_size": bs,
                "input_length": ilen,
                "input_sequence_hashes": hashes,
            },
        },
    }


def _multi_tree_records() -> list[dict]:
    """Three session-trees with DISJOINT content/sessions.

    * ``aaa``: root (2 turns) + subagent ``bbb`` (within-tree parent<->child edge)
    * ``ccc``: root (2 turns)
    * ``ddd``: root (single turn)

    Roots sort ``aaa < ccc < ddd``, so the first tree (``ParsedGraph.graph``
    back-compat) is the ``aaa`` tree.
    """
    return [
        _re(ts=1000, sid="aaa", hashes=[1, 2], ilen=32),
        _re(ts=1500, sid="bbb", parent="aaa", hashes=[7, 8], ilen=32),
        _re(ts=2000, sid="aaa", hashes=[1, 2, 3], ilen=48),
        _re(ts=5000, sid="ccc", hashes=[20, 21], ilen=32),
        _re(ts=5100, sid="ccc", hashes=[20, 21, 22], ilen=48),
        _re(ts=9000, sid="ddd", hashes=[30, 31], ilen=32),
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def _fixture(tmp_path: Path) -> Path:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _multi_tree_records())
    return p


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()


def _edge_key(edge) -> str:
    return (
        f"{edge.source}->{edge.target}"
        f"|dap={edge.delay_after_predecessor_us}"
        f"|min={edge.min_start_delay_us}"
        f"|daps={edge.delay_after_predecessor_start_us}"
        f"|dapft={edge.delay_after_predecessor_first_token_us}"
    )


def _node_session(node: LlmNode) -> str:
    return node.metadata["dynamo"]["session_id"]


def _parse(path: Path):
    return from_dynamo_trace(
        path, content_root_seed=_SEED, content_tokenizer=_TOKENIZER
    )


# --- 1. CONTENT ORACLE: union(new per-tree graphs) == pre-change union ------


def test_reunion_of_per_tree_graphs_matches_pinned_single_union(
    tmp_path: Path,
) -> None:
    """union(list(parsed.graphs.values())) reproduces the pinned pre-change union.

    The retained test-only ``_union_graphs`` re-flattens the per-tree graphs;
    since those graphs ARE the graphs the old path unioned, the nodes, edges,
    and content-addressed pool keys are byte-identical to the single union
    captured before the restructure.
    """
    parsed = _parse(_fixture(tmp_path))

    assert parsed.graphs, "multi-graph shape required (graphs must be populated)"
    union = _union_graphs(list(parsed.graphs.values()))

    assert _digest("\n".join(sorted(union.nodes))) == _ORACLE_NODES_DIGEST
    assert (
        _digest("\n".join(sorted(_edge_key(e) for e in union.edges)))
        == _ORACLE_EDGES_DIGEST
    )
    assert parsed.segment_pool is not None
    assert _digest("\n".join(sorted(parsed.segment_pool.by_id))) == _ORACLE_POOL_DIGEST


def test_union_node_and_edge_counts_unchanged(tmp_path: Path) -> None:
    """The re-union has the same 6 nodes / 6 edges the pre-change union had."""
    parsed = _parse(_fixture(tmp_path))
    union = _union_graphs(list(parsed.graphs.values()))
    assert len(union.nodes) == 6
    assert len(union.edges) == 6


# --- 2. SCOPING: per-trace graph_ref + tree-scoped resolution ---------------


def test_each_trace_is_its_own_single_root_tree(tmp_path: Path) -> None:
    parsed = _parse(_fixture(tmp_path))

    # One trace per tree, id-sorted, each keyed into graphs by its root id.
    assert [t.id for t in parsed.traces] == ["aaa", "ccc", "ddd"]
    for trace in parsed.traces:
        assert trace.graph_ref == trace.id, (
            f"trace {trace.id!r} graph_ref must equal its root session id"
        )
        assert trace.graph_ref in parsed.graphs
        # multi-root tag is dropped: each tree is its own single-root trace.
        assert "multi-root" not in trace.tags
        assert "from-dynamo-trace" in trace.tags


def test_resolve_trace_graph_returns_only_that_trees_nodes(tmp_path: Path) -> None:
    parsed = _parse(_fixture(tmp_path))

    expected_sessions = {"aaa": {"aaa", "bbb"}, "ccc": {"ccc"}, "ddd": {"ddd"}}
    for trace in parsed.traces:
        graph = resolve_trace_graph(parsed, trace)
        sessions = {_node_session(n) for n in graph.nodes.values()}
        assert sessions == expected_sessions[trace.id], (
            f"trace {trace.id!r} resolved to sessions {sessions}"
        )


def test_two_distinct_traces_resolve_to_disjoint_node_sets(tmp_path: Path) -> None:
    parsed = _parse(_fixture(tmp_path))
    by_id = {t.id: t for t in parsed.traces}

    aaa_nodes = set(resolve_trace_graph(parsed, by_id["aaa"]).nodes)
    ccc_nodes = set(resolve_trace_graph(parsed, by_id["ccc"]).nodes)
    assert aaa_nodes and ccc_nodes
    assert aaa_nodes.isdisjoint(ccc_nodes)


def test_back_compat_graph_is_first_tree_not_union(tmp_path: Path) -> None:
    """``ParsedGraph.graph`` is the FIRST tree (lex-min root ``aaa``), not the union."""
    parsed = _parse(_fixture(tmp_path))
    first_sessions = {_node_session(n) for n in parsed.graph.nodes.values()}
    assert first_sessions == {"aaa", "bbb"}
    assert len(parsed.graph.nodes) < 6  # strictly fewer than the union's 6


# --- 3. fused-parallel path emits the SAME multi-graph shape ----------------


def test_fused_parallel_emits_same_multigraph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forcing the pool path (per-tree ParsedGraph blobs from workers) yields the
    identical multi-graph shape + content as the serial per-tree build."""
    path = _fixture(tmp_path)
    serial = _parse(path)

    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 0)
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_WORKERS", 2)
    parallel = _parse(path)

    assert [t.id for t in parallel.traces] == [t.id for t in serial.traces]
    assert {t.graph_ref for t in parallel.traces} == {"aaa", "ccc", "ddd"}
    for tid in ("aaa", "ccc", "ddd"):
        assert set(parallel.graphs[tid].nodes) == set(serial.graphs[tid].nodes)
    p_union = _union_graphs(list(parallel.graphs.values()))
    assert _digest("\n".join(sorted(p_union.nodes))) == _ORACLE_NODES_DIGEST
    assert parallel.segment_pool is not None
    assert parallel.segment_pool.by_id == serial.segment_pool.by_id
