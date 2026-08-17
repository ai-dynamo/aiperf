# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Format detection for Dynamo agent traces: which .jsonl/.jsonl.gz/segmented-dir shapes the adapter claims, and which it must leave alone."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import DynamoTraceAdapter
from aiperf.dataset.graph.parser import (
    WorkloadFormatError,
    detect_format,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


def dynamo_record(
    *,
    event_type: str = "request_end",
    agent_context: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """One well-formed dynamo.request.trace.v1 event."""
    rec: dict[str, Any] = {
        "schema": "dynamo.request.trace.v1",
        "event_type": event_type,
        "agent_context": {} if agent_context is None else agent_context,
        "ts": "2026-05-06T00:00:00.000Z",
    }
    rec.update(overrides)
    return rec


def write_jsonl(path: Path, recs: list[dict[str, Any]]) -> None:
    """Write records as plain newline-delimited JSON."""
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def write_jsonl_gz(path: Path, recs: list[dict[str, Any]]) -> None:
    """Write records as gzip-compressed newline-delimited JSON."""
    payload = ("\n".join(json.dumps(r) for r in recs) + "\n").encode()
    path.write_bytes(gzip.compress(payload))


# A plausible non-dynamo request-trace doc, keyed on
# {id, models, block_size, hash_id_scope, requests}.
_FOREIGN_TRACE_DOC: dict[str, Any] = {
    "id": "trace_ordering_guard",
    "models": ["m"],
    "block_size": 64,
    "hash_id_scope": "local",
    "requests": [{"t": 0.0, "type": "n", "model": "m", "in": 10, "out": 5}],
}

_NATIVE_GRAPH_DOC: dict[str, Any] = {
    "graph": {
        "nodes": {
            "a": {
                "node_type": "llm",
                "prompt": [{"role": "user", "content": "hi"}],
                "output": "out",
            }
        },
        "edges": [],
    },
    "traces": [{"id": "t1"}],
}

# Valid 10-byte gzip header + non-deflate payload -> zlib.error on read.
_GZ_HEADER_PLUS_GARBAGE = (
    b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"this-is-not-deflate-data"
)


def test_graph_adapter_registry_includes_dynamo_trace() -> None:
    """The dynamo_trace adapter is registered, so detection can reach it at all."""
    assert plugins.has_entry(PluginType.GRAPH_ADAPTER, "dynamo_trace")


class TestClaimedLayouts:
    """Captures that detect_format must route to dynamo_trace."""

    def test_single_jsonl_file(self, tmp_path: Path) -> None:
        p = tmp_path / "trace.jsonl"
        write_jsonl(p, [dynamo_record(), dynamo_record(event_type="tool_start")])
        assert detect_format(p) == "dynamo_trace"

    def test_single_gzipped_jsonl_file(self, tmp_path: Path) -> None:
        p = tmp_path / "trace.jsonl.gz"
        write_jsonl_gz(
            p,
            [
                dynamo_record(),
                dynamo_record(event_type="tool_start"),
                dynamo_record(event_type="tool_end"),
            ],
        )
        assert detect_format(p) == "dynamo_trace"

    def test_segmented_gzipped_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "agent-traces"
        d.mkdir()
        write_jsonl_gz(d / "agent.000000.jsonl.gz", [dynamo_record()])
        write_jsonl_gz(
            d / "agent.000001.jsonl.gz", [dynamo_record(event_type="tool_start")]
        )
        assert detect_format(d) == "dynamo_trace"

    def test_plain_jsonl_dir(self, tmp_path: Path) -> None:
        """Un-gzipped capture dirs are claimed too, since discover_segments also globs *.jsonl."""
        d = tmp_path / "agent-traces-plain"
        d.mkdir()
        write_jsonl(d / "a.jsonl", [dynamo_record()])
        write_jsonl(d / "b.jsonl", [dynamo_record(event_type="tool_end")])
        assert detect_format(d) == "dynamo_trace"

    def test_empty_agent_context(self, tmp_path: Path) -> None:
        """The predicate only requires agent_context to be a dict, so an empty one still routes."""
        p = tmp_path / "trace.jsonl"
        write_jsonl(p, [dynamo_record(agent_context={})])
        assert detect_format(p) == "dynamo_trace"


class TestFallThrough:
    """Captures detection must NOT claim; dynamo_trace is the only registered adapter here, so each falls through to WorkloadFormatError."""

    @pytest.mark.parametrize(
        "record",
        [
            param(
                {
                    "schema": "other.schema.v1",
                    "event_type": "request_end",
                    "agent_context": {},
                },
                id="foreign_schema",
            ),
            param(
                {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "stream_chunk",
                    "agent_context": {},
                },
                id="unknown_event_type",
            ),
        ],
    )  # fmt: skip
    def test_jsonl_record_not_matching_predicate(
        self, tmp_path: Path, record: dict[str, Any]
    ) -> None:
        p = tmp_path / "other.jsonl"
        write_jsonl(p, [record])
        with pytest.raises(WorkloadFormatError):
            detect_format(p)

    def test_plain_jsonl_dir_with_foreign_records(self, tmp_path: Path) -> None:
        """Foreign .jsonl dirs fall through: the dir branch still sniffs the first record instead of claiming by extension alone."""
        d = tmp_path / "not-dynamo"
        d.mkdir()
        write_jsonl(
            d / "a.jsonl", [{"schema": "other.v1", "event_type": "request_end"}]
        )
        with pytest.raises(WorkloadFormatError):
            detect_format(d)

    def test_directory_with_no_trace_files(self, tmp_path: Path) -> None:
        d = tmp_path / "mystery-dir"
        d.mkdir()
        (d / "unrelated.txt").write_text("hello")
        with pytest.raises(WorkloadFormatError):
            detect_format(d)

    def test_crashing_adapter_sniff_is_contained(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One adapter's can_load raising is reported as "unrecognized" rather than propagating, so it cannot abort detection for the rest."""

        def _boom(cls: type, path: Path) -> bool:
            raise RuntimeError("sniff crashed")

        monkeypatch.setattr(DynamoTraceAdapter, "can_load", classmethod(_boom))
        p = tmp_path / "some.jsonl"
        write_jsonl(p, [{"kind": "note", "text": "not a dynamo record"}])
        with pytest.raises(WorkloadFormatError):
            detect_format(p)


class TestCanLoadPredicate:
    """DynamoTraceAdapter.can_load must answer False rather than crash or over-claim."""

    @pytest.mark.parametrize(
        ("filename", "writer"),
        [
            param(
                "trace.jsonl.gz",
                lambda p: p.write_bytes(_GZ_HEADER_PLUS_GARBAGE),
                id="corrupt_deflate_payload",
            ),
            param(
                "trace.jsonl.gz",
                lambda p: (
                    write_jsonl_gz(p, [dynamo_record()]),
                    p.write_bytes(p.read_bytes()[:20]),
                ),
                id="truncated_gzip",
            ),
            param(
                "trace_01.json",
                lambda p: p.write_text(json.dumps(_FOREIGN_TRACE_DOC)),
                id="foreign_trace_doc",
            ),
            param(
                "workload.jsonl",
                lambda p: write_jsonl(p, [_NATIVE_GRAPH_DOC]),
                id="native_graph_jsonl",
            ),
        ],
    )  # fmt: skip
    def test_can_load_returns_false(
        self, tmp_path: Path, filename: str, writer: Any
    ) -> None:
        # dynamo_trace holds the highest detection_priority (100), so an
        # over-claiming predicate would win every tie-break and silently reroute
        # foreign workloads -- including native graph docs, which share .jsonl.
        # Corrupt/truncated gzip must answer "not ours", never raise zlib.error.
        p = tmp_path / filename
        writer(p)
        assert DynamoTraceAdapter.can_load(p) is False


def test_parser_dispatch_routes_to_from_dynamo_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """parse_graph dispatches a detected dynamo_trace path into from_dynamo_trace with that same path."""
    from aiperf.dataset.graph import parser as parser_mod
    from aiperf.dataset.graph.adapters.dynamo import trace as dt_mod
    from aiperf.dataset.graph.models import GraphRecord, ParsedGraph

    p = tmp_path / "trace.jsonl"
    write_jsonl(p, [dynamo_record()])

    called: dict[str, Any] = {}

    def _fake_from_dynamo_trace(path: Path, **kwargs: Any) -> ParsedGraph:
        called["path"] = Path(path)
        return ParsedGraph(graph=GraphRecord(), traces=[])

    monkeypatch.setattr(dt_mod, "from_dynamo_trace", _fake_from_dynamo_trace)
    parser_mod.parse_graph(p)

    assert called["path"] == p
