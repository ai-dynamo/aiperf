# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""graph_meta sidecar writer and its reuse gates: catalog match, store-index coverage, and replay-text stripping."""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.graph.codecs import decode_graph_meta_sidecar
from aiperf.dataset.graph.graph_meta_sidecar import (
    catalogs_match,
    sidecar_matches_index,
    sidecar_path_for,
    strip_replay_text,
    write_graph_meta_sidecar,
)
from aiperf.dataset.graph.graph_path_catalog import build_catalog_context
from aiperf.dataset.graph.models import GraphRecord, LlmNode, ParsedGraph, TraceRecord


def node_free_graph() -> ParsedGraph:
    """A one-trace graph with no nodes, whose catalog is therefore empty."""
    return ParsedGraph(graph=GraphRecord(), traces=[TraceRecord(id="t-1", tags=["x"])])


def graph_with_node() -> ParsedGraph:
    """A one-trace graph with one LlmNode, so its catalog carries a real node ordinal."""
    # A node-free graph makes every ordinal comparison vacuously True, which
    # would hide the index-coverage logic entirely.
    graph = GraphRecord(nodes={"n0": LlmNode(prompt=["hi"], output="out")}, edges=[])
    return ParsedGraph(graph=graph, traces=[TraceRecord(id="t-1", tags=["x"])])


def test_write_then_decode_round_trips(tmp_path: Path) -> None:
    """The writer lands at sidecar_path_for() and the bytes decode back to the same traces, fingerprint, and version."""
    out = write_graph_meta_sidecar(
        node_free_graph(),
        base_path=tmp_path,
        benchmark_id="bench-9",
        source_fingerprint={"kind": "file"},
        schema_version=1,
    )
    assert out == sidecar_path_for(tmp_path, "bench-9")
    assert out.exists()
    decoded, fp, version = decode_graph_meta_sidecar(out.read_bytes())
    assert [t.id for t in decoded.traces] == ["t-1"]
    assert fp == {"kind": "file"} and version == 1


def test_catalogs_match_true_for_same_graph() -> None:
    """A catalog rebuilt from the same graph matches it."""
    pg = node_free_graph()
    assert catalogs_match(pg, build_catalog_context(pg).catalog) is True


def test_catalogs_match_false_for_divergent_catalog() -> None:
    """A catalog naming a trace the graph does not contain does not match."""
    assert catalogs_match(node_free_graph(), {"ghost-trace": {"n": 0}}) is False


def test_sidecar_matches_index_true_when_store_covers_catalog() -> None:
    """A store index whose per-trace ordinals cover every catalog ordinal is reusable."""
    pg = graph_with_node()
    catalog = build_catalog_context(pg).catalog
    assert catalog["t-1"], "fixture must yield real node ordinals"
    index = {t: {o: None for o in ords.values()} for t, ords in catalog.items()}
    assert sidecar_matches_index(pg, index) is True


def test_sidecar_matches_index_false_when_catalog_ordinal_missing_from_store() -> None:
    """A catalog ordinal absent from the store means topology drift, so the sidecar must be rejected and the graph re-parsed."""
    pg = graph_with_node()
    assert sidecar_matches_index(pg, {"t-1": {}}) is False
    assert sidecar_matches_index(pg, {}) is False


def test_strip_replay_text_clears_only_replay_outputs() -> None:
    """Stripping drops replay_outputs while preserving id/tags, and leaves the input ParsedGraph untouched."""
    tr = TraceRecord(
        id="t-1",
        tags=["x"],
        replay_outputs={
            "n__msgdelta": {"messages": [{"role": "user", "content": "hi"}]}
        },
    )
    pg = ParsedGraph(graph=GraphRecord(), traces=[tr])
    stripped = strip_replay_text(pg)
    assert stripped.traces[0].replay_outputs == {}
    assert stripped.traces[0].id == "t-1"
    assert stripped.traces[0].tags == ["x"]
    assert pg.traces[0].replay_outputs != {}
