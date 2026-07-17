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


def _graph() -> ParsedGraph:
    return ParsedGraph(graph=GraphRecord(), traces=[TraceRecord(id="t-1", tags=["x"])])


def _graph_with_node() -> ParsedGraph:
    """A graph whose catalog carries a REAL node ordinal.

    A node-free graph yields an empty per-trace catalog, which makes every
    ``sidecar_matches_index`` comparison vacuously True -- the False branch
    needs an ordinal that CAN go missing from the store index.
    """
    graph = GraphRecord(nodes={"n0": LlmNode(prompt=["hi"], output="out")}, edges=[])
    return ParsedGraph(graph=graph, traces=[TraceRecord(id="t-1", tags=["x"])])


def test_write_then_decode_round_trips(tmp_path: Path):
    pg = _graph()
    out = write_graph_meta_sidecar(
        pg,
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


def test_catalogs_match_true_for_same_graph():
    pg = _graph()
    catalog = build_catalog_context(pg).catalog
    assert catalogs_match(pg, catalog) is True


def test_catalogs_match_false_for_divergent_catalog():
    pg = _graph()
    assert catalogs_match(pg, {"ghost-trace": {"n": 0}}) is False


def test_sidecar_matches_index_true_when_store_covers_catalog():
    pg = _graph_with_node()
    catalog = build_catalog_context(pg).catalog
    assert catalog["t-1"], "fixture must yield real node ordinals"
    # Build a fake index whose per-trace integer ordinals cover the catalog.
    index = {
        t: {(o, "profiling"): None for o in ords.values()}
        for t, ords in catalog.items()
    }
    assert sidecar_matches_index(pg, index) is True


def test_sidecar_matches_index_false_when_catalog_ordinal_missing_from_store():
    """The False branch is the function's reason to exist: a catalog ordinal
    absent from the store index means topology drift -> fall back to re-parse."""
    pg = _graph_with_node()
    assert sidecar_matches_index(pg, {"t-1": {}}) is False
    assert sidecar_matches_index(pg, {}) is False


def test_strip_replay_text_clears_only_replay_outputs():
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
    # original must be untouched
    assert pg.traces[0].replay_outputs != {}
