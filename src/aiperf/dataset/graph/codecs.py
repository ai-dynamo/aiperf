# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached msgspec codecs for the agent graph (canonical struct (de)serialization).

Decoders snapshot their target type at first use, so they are module-level and
built once. msgpack is the cross-process store format.
"""

from __future__ import annotations

from typing import Any

import msgspec

from aiperf.dataset.graph.models import ParsedGraph

_PG_MSGPACK_ENCODER = msgspec.msgpack.Encoder()
_PG_MSGPACK_DECODER = msgspec.msgpack.Decoder(ParsedGraph)


def encode_parsed_graph_msgpack(pg: ParsedGraph) -> bytes:
    """Encode a :class:`ParsedGraph` to canonical msgpack bytes (cross-process)."""
    return _PG_MSGPACK_ENCODER.encode(pg)


def decode_parsed_graph_msgpack(data: bytes) -> ParsedGraph:
    """Decode canonical msgpack bytes back into a :class:`ParsedGraph`."""
    return _PG_MSGPACK_DECODER.decode(data)


GRAPH_META_SIDECAR_FILENAME = "graph_meta.msgpack"

# Sidecar frame ``kind`` discriminator: every frame carries a node-typed
# ``ParsedGraph`` and MUST declare ``kind`` explicitly (see the strict decode).
_SIDECAR_KIND_PARSED_GRAPH = "parsed_graph"

# Bumped 2->3 when the kind-less decode default was retired: the decoder now
# REQUIRES an explicit ``kind`` header, so old persisted sidecars/cache entries
# (written without it) version-gate out and self-heal on the next run.
# Bumped 3->4 with the verbatim raw-JSON segment variant (``Segment.wire_json``):
# pre-v4 stores normalized every segment to ``{"role", "content"}`` (dropping key
# order and extra keys). Unlike the 2->3 bump, v4 adds NO reader-side gate: the
# decoder tolerates a pre-v4 blob because ``Segment.wire_json`` defaults to ``None``
# (a normalized role/content segment), so the version here is advisory provenance,
# not a compatibility gate. msgspec auto-encodes the new dataclass field, so the
# cross-process ``ParsedGraph`` codec round-trips it without an explicit hook.
GRAPH_META_SCHEMA_VERSION: int = 4

_SIDECAR_DECODER = msgspec.msgpack.Decoder()


def encode_graph_meta_sidecar(
    pg: ParsedGraph,
    *,
    source_fingerprint: dict[str, Any],
    schema_version: int = GRAPH_META_SCHEMA_VERSION,
) -> bytes:
    """Encode a content-free structural ``ParsedGraph`` + identity header to bytes.

    The frame is ``[header, pg_bytes]``; ``pg_bytes`` is the canonical
    ``encode_parsed_graph_msgpack`` output kept as a nested blob so the existing
    typed decoder round-trips the graph unchanged. The header declares an explicit
    ``kind`` so the strict decoder can reject kind-less pre-v3 artifacts.
    """
    header = {
        "kind": _SIDECAR_KIND_PARSED_GRAPH,
        "schema_version": schema_version,
        "source_fingerprint": source_fingerprint,
    }
    return _PG_MSGPACK_ENCODER.encode([header, encode_parsed_graph_msgpack(pg)])


def decode_graph_meta_sidecar(
    data: bytes,
) -> tuple[ParsedGraph, dict[str, Any], int]:
    """Decode a sidecar frame into ``(parsed_graph, source_fingerprint, schema_version)``.

    The blob is a content-free node-typed :class:`ParsedGraph`. The header ``kind``
    is REQUIRED and must equal ``"parsed_graph"``: a kind-less frame (a pre-v3
    persisted artifact) or an unsupported kind raises ``ValueError``; the
    TimingManager surfaces it as ``InvalidStateError`` (the sidecar is mandatory;
    no re-parse fallback exists).

    Raises ``msgspec.DecodeError`` / ``ValueError`` on a malformed frame.
    """
    frame = _SIDECAR_DECODER.decode(data)
    if not isinstance(frame, list) or len(frame) != 2:
        raise ValueError("graph_meta sidecar frame must be [header, blob]")
    header, blob = frame
    if not isinstance(header, dict) or not isinstance(blob, bytes):
        raise ValueError(
            "graph_meta sidecar frame has the wrong shape: expected header "
            f"dict and blob bytes, got {type(header).__name__} and "
            f"{type(blob).__name__}; rebuild the graph store"
        )
    if "source_fingerprint" not in header or "schema_version" not in header:
        raise ValueError("graph_meta sidecar header missing required keys")
    if header.get("kind") != _SIDECAR_KIND_PARSED_GRAPH:
        raise ValueError(
            "unsupported or missing graph_meta sidecar kind; rebuild the graph store"
        )
    return (
        decode_parsed_graph_msgpack(blob),
        header["source_fingerprint"],
        int(header["schema_version"]),
    )
