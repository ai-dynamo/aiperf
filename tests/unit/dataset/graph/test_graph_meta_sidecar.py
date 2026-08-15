# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph metadata sidecars round-trip, and malformed mandatory sidecars fail."""

from __future__ import annotations

import msgspec
import pytest
from pytest import param

from aiperf.dataset.graph.codecs import (
    GRAPH_META_SCHEMA_VERSION,
    GRAPH_META_SIDECAR_FILENAME,
    decode_graph_meta_sidecar,
    encode_graph_meta_sidecar,
)
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph, TraceRecord


def tiny_graph() -> ParsedGraph:
    """The smallest ParsedGraph the sidecar codec accepts: one tagged trace, empty graph."""
    return ParsedGraph(
        graph=GraphRecord(),
        traces=[TraceRecord(id="t-1", tags=["x"])],
    )


def test_sidecar_round_trips_graph_and_header() -> None:
    """Encoding then decoding restores the schema version, the source fingerprint, and the traces."""
    fp = {"kind": "file", "sha256": "abc", "size": 12}
    blob = encode_graph_meta_sidecar(
        tiny_graph(), source_fingerprint=fp, schema_version=1
    )
    decoded, decoded_fp, version = decode_graph_meta_sidecar(blob)
    assert version == 1
    assert decoded_fp == fp
    assert [t.id for t in decoded.traces] == ["t-1"]


def test_encoder_writes_explicit_kind_and_decoder_requires_it() -> None:
    """The header carries an explicit kind/schema_version, and kind-less pre-v3 frames are rejected."""
    frame = encode_graph_meta_sidecar(tiny_graph(), source_fingerprint={"k": "v"})
    header, blob = msgspec.msgpack.decode(frame)
    assert header["kind"] == "parsed_graph"
    assert header["schema_version"] == GRAPH_META_SCHEMA_VERSION

    del header["kind"]
    with pytest.raises(ValueError, match="kind"):
        decode_graph_meta_sidecar(msgspec.msgpack.encode([header, blob]))


def test_sidecar_filename_constant() -> None:
    """The sidecar filename is part of the on-disk contract with readers."""
    assert GRAPH_META_SIDECAR_FILENAME == "graph_meta.msgpack"


@pytest.mark.parametrize(
    "frame",
    [
        param(b"\xff\xff not msgpack", id="not_msgpack_at_all"),
        param(msgspec.msgpack.encode("not-a-list"), id="parseable_but_wrong_shape"),
        param(
            msgspec.msgpack.encode([{"schema_version": 1}, b"ignored"]),
            id="header_missing_required_keys",
        ),
    ],
)  # fmt: skip
def test_decode_rejects_malformed_frame(frame: bytes) -> None:
    """Anything other than a valid frame is rejected without a parse fallback."""
    with pytest.raises((msgspec.DecodeError, ValueError)):
        decode_graph_meta_sidecar(frame)


@pytest.mark.parametrize(
    ("frame", "actual_type"),
    [
        param(msgspec.msgpack.encode([["header"], b"blob"]), "list", id="header"),
        param(
            msgspec.msgpack.encode(
                [
                    {
                        "kind": "parsed_graph",
                        "schema_version": GRAPH_META_SCHEMA_VERSION,
                        "source_fingerprint": {},
                    },
                    "blob",
                ]
            ),
            "str",
            id="blob",
        ),
    ],
)  # fmt: skip
def test_decode_wrong_element_type_reports_shape_and_remediation(
    frame: bytes, actual_type: str
) -> None:
    """Wrong element types identify the observed shape and recovery action."""
    with pytest.raises(ValueError) as exc_info:
        decode_graph_meta_sidecar(frame)

    message = str(exc_info.value)
    assert "expected header dict and blob bytes" in message
    assert actual_type in message
    assert "rebuild the graph store" in message
