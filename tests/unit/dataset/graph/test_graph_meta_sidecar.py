import msgspec
import pytest

from aiperf.dataset.graph.codecs import (
    GRAPH_META_SCHEMA_VERSION,
    GRAPH_META_SIDECAR_FILENAME,
    decode_graph_meta_sidecar,
    encode_graph_meta_sidecar,
)
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph, TraceRecord


def _tiny_graph() -> ParsedGraph:
    return ParsedGraph(
        graph=GraphRecord(),
        traces=[TraceRecord(id="t-1", tags=["x"])],
    )


def test_sidecar_round_trips_graph_and_header():
    pg = _tiny_graph()
    fp = {"kind": "file", "sha256": "abc", "size": 12}
    blob = encode_graph_meta_sidecar(pg, source_fingerprint=fp, schema_version=1)
    decoded, decoded_fp, version = decode_graph_meta_sidecar(blob)
    assert version == 1
    assert decoded_fp == fp
    assert [t.id for t in decoded.traces] == ["t-1"]


def test_encoder_writes_explicit_kind_and_decoder_requires_it() -> None:
    pg = _tiny_graph()
    frame = encode_graph_meta_sidecar(pg, source_fingerprint={"k": "v"})
    header, _blob = msgspec.msgpack.decode(frame)
    assert header["kind"] == "parsed_graph"
    assert header["schema_version"] == GRAPH_META_SCHEMA_VERSION

    # Kind-less frames (pre-v3 artifacts) are rejected -> caller re-parses.
    del header["kind"]
    stale = msgspec.msgpack.encode([header, _blob])
    with pytest.raises(ValueError, match="kind"):
        decode_graph_meta_sidecar(stale)


def test_sidecar_filename_constant():
    assert GRAPH_META_SIDECAR_FILENAME == "graph_meta.msgpack"


def test_decode_rejects_garbage():
    with pytest.raises((msgspec.DecodeError, ValueError)):
        decode_graph_meta_sidecar(b"\xff\xff not msgpack")


def test_decode_rejects_wrong_shape_frame():
    # Parseable msgpack, but not the [header, pg_bytes] frame shape.
    with pytest.raises(ValueError):
        decode_graph_meta_sidecar(msgspec.msgpack.encode("not-a-list"))


def test_decode_rejects_header_missing_keys():
    frame = msgspec.msgpack.encode([{"schema_version": 1}, b"ignored"])
    with pytest.raises(ValueError):
        decode_graph_meta_sidecar(frame)
