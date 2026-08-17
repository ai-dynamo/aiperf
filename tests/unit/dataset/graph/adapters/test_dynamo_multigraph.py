# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo multi-graph restructure: one per-tree ``GraphRecord`` selected by ``TraceRecord.graph_ref``."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec.structs
import pytest

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.adapters.shared.content import CorpusContentSynthesizer
from aiperf.dataset.graph.models import LlmNode, ParsedGraph, resolve_trace_graph
from tests.unit.dataset.graph.adapters.conftest import blake_digest, write_jsonl

_SEED = 20260706
_TOKENIZER = "builtin"

# The dynamo adapter emits ONE GraphRecord per session-tree (mirroring the weka
# multi-item path through merge_parsed_graphs): each trace owns its top-level
# graph under ParsedGraph.graphs[root_session_id] and its graph_ref selects it.
# The whole-capture union is no longer the emitted shape -- ParsedGraph.graph is
# now the FIRST tree (back-compat), not the union. On the pre-change
# single-union code this module goes RED: parsed.graphs is empty and graph_ref is
# None, so the scoping asserts fail.

# Pinned segment-pool oracle over the fixture below. A drift here is a real
# content change, not a shape change -- investigate, do not re-pin blindly.
# Re-pinned 2026-07-07 for the data-inherent node-id scheme: node ids became
# {session_id}:{k} (recorded session id verbatim, 0-based turn), and the
# trace-scoped response/tiny synthesis seeds shift the node-id-seeded pool
# entries. Hash-block prompt content is unchanged; the determinism guard (two
# independent builds agree) still holds.
_ORACLE_POOL_DIGEST = "f963d9998288d2d9fd1c66cd52844ad9"

# Sessions belonging to each tree's root, keyed by root/trace id.
_TREE_SESSIONS = {"aaa": {"aaa", "bbb"}, "ccc": {"ccc"}, "ddd": {"ddd"}}


@pytest.fixture(autouse=True)
def _fresh_synth_cache() -> Iterator[None]:
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
) -> dict[str, Any]:
    """A ``request_end`` record in the shape real dynamo captures emit."""
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


def _multi_tree_records() -> list[dict[str, Any]]:
    """Three DISJOINT session-trees: 'aaa' (2 turns + subagent 'bbb'), 'ccc' (2 turns), 'ddd' (1 turn)."""
    # Roots sort aaa < ccc < ddd, so the first tree (ParsedGraph.graph) is aaa's.
    return [
        _re(ts=1000, sid="aaa", hashes=[1, 2], ilen=32),
        _re(ts=1500, sid="bbb", parent="aaa", hashes=[7, 8], ilen=32),
        _re(ts=2000, sid="aaa", hashes=[1, 2, 3], ilen=48),
        _re(ts=5000, sid="ccc", hashes=[20, 21], ilen=32),
        _re(ts=5100, sid="ccc", hashes=[20, 21, 22], ilen=48),
        _re(ts=9000, sid="ddd", hashes=[30, 31], ilen=32),
    ]


def _fixture(tmp_path: Path) -> Path:
    """Write the three-tree capture and return its path."""
    return write_jsonl(tmp_path / "trace.jsonl", _multi_tree_records())


def _node_session(node: LlmNode) -> str:
    """The recorded dynamo session id a lowered node came from."""
    return node.metadata["dynamo"]["session_id"]


def _parse(path: Path) -> ParsedGraph:
    """Parse a capture with the pinned oracle seed and tokenizer."""
    return from_dynamo_trace(
        path, content_root_seed=_SEED, content_tokenizer=_TOKENIZER
    )


# --- 1. CONTENT ORACLE: segment pool pinned across the multi-graph shape ----


def test_segment_pool_matches_pinned_oracle(tmp_path: Path) -> None:
    """The synthesized segment pool is byte-stable across the multi-graph restructure."""
    parsed = _parse(_fixture(tmp_path))

    assert parsed.graphs, "multi-graph shape required (graphs must be populated)"
    assert parsed.segment_pool is not None
    assert (
        blake_digest("\n".join(sorted(parsed.segment_pool.by_id)))
        == _ORACLE_POOL_DIGEST
    )


# --- 2. SCOPING: per-trace graph_ref + tree-scoped resolution ---------------


def test_each_trace_is_its_own_single_root_tree(tmp_path: Path) -> None:
    """One id-sorted trace per tree, each keyed into ``graphs`` by its root session id and no longer tagged multi-root."""
    parsed = _parse(_fixture(tmp_path))

    assert [t.id for t in parsed.traces] == ["aaa", "ccc", "ddd"]
    for trace in parsed.traces:
        assert trace.graph_ref == trace.id, (
            f"trace {trace.id!r} graph_ref must equal its root session id"
        )
        assert trace.graph_ref in parsed.graphs
        assert "multi-root" not in trace.tags
        assert "from-dynamo-trace" in trace.tags


def test_resolve_trace_graph_returns_only_that_trees_nodes(tmp_path: Path) -> None:
    """``resolve_trace_graph`` scopes each trace to exactly its own tree's sessions."""
    parsed = _parse(_fixture(tmp_path))

    for trace in parsed.traces:
        graph = resolve_trace_graph(parsed, trace)
        sessions = {_node_session(n) for n in graph.nodes.values()}
        assert sessions == _TREE_SESSIONS[trace.id], (
            f"trace {trace.id!r} resolved to sessions {sessions}"
        )


def test_dangling_graph_ref_warns_and_falls_back_to_default_graph(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A ``graph_ref`` absent from ``parsed.graphs`` degrades, it does not KeyError."""
    parsed = _parse(_fixture(tmp_path))
    trace = msgspec.structs.replace(parsed.traces[0], graph_ref="no-such-tree")

    with caplog.at_level("WARNING"):
        graph = resolve_trace_graph(parsed, trace)

    assert graph is parsed.graph
    assert "no-such-tree" in caplog.text


def test_two_distinct_traces_resolve_to_disjoint_node_sets(tmp_path: Path) -> None:
    """Two distinct traces resolve to non-empty, mutually disjoint node sets."""
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
    """The pool path (per-tree ParsedGraph blobs from workers) yields the identical multi-graph shape and content as the serial per-tree build."""
    path = _fixture(tmp_path)
    serial = _parse(path)

    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 0)
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_WORKERS", 2)
    parallel = _parse(path)

    assert [t.id for t in parallel.traces] == [t.id for t in serial.traces]
    assert {t.graph_ref for t in parallel.traces} == set(_TREE_SESSIONS)
    for tid in _TREE_SESSIONS:
        assert set(parallel.graphs[tid].nodes) == set(serial.graphs[tid].nodes)
    assert parallel.segment_pool is not None
    assert parallel.segment_pool.by_id == serial.segment_pool.by_id
