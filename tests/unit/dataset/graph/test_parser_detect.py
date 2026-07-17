# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Format-detection tests for Dynamo agent-trace JSONL/JSONL.gz/segmented dirs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from aiperf.dataset.graph.parser import (
    WorkloadFormatError,
    detect_format,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


def _dynamo_record(
    *,
    event_type: str = "request_end",
    agent_context: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "schema": "dynamo.request.trace.v1",
        "event_type": event_type,
        "agent_context": {} if agent_context is None else agent_context,
        "ts": "2026-05-06T00:00:00.000Z",
    }
    rec.update(overrides)
    return rec


def _write_jsonl(path: Path, recs: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def _write_jsonl_gz(path: Path, recs: list[dict[str, Any]]) -> None:
    payload = ("\n".join(json.dumps(r) for r in recs) + "\n").encode()
    path.write_bytes(gzip.compress(payload))


def test_graph_adapter_registry_includes_dynamo_trace() -> None:
    assert plugins.has_entry(PluginType.GRAPH_ADAPTER, "dynamo_trace")


def test_detects_dynamo_trace_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, [_dynamo_record(), _dynamo_record(event_type="tool_start")])
    assert detect_format(p) == "dynamo_trace"


def test_detects_dynamo_trace_jsonl_gz(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl.gz"
    _write_jsonl_gz(
        p,
        [
            _dynamo_record(),
            _dynamo_record(event_type="tool_start"),
            _dynamo_record(event_type="tool_end"),
        ],
    )
    assert detect_format(p) == "dynamo_trace"


def test_detects_segmented_dynamo_trace_dir(tmp_path: Path) -> None:
    d = tmp_path / "agent-traces"
    d.mkdir()
    _write_jsonl_gz(d / "agent.000000.jsonl.gz", [_dynamo_record()])
    _write_jsonl_gz(
        d / "agent.000001.jsonl.gz", [_dynamo_record(event_type="tool_start")]
    )
    assert detect_format(d) == "dynamo_trace"


def test_detects_plain_jsonl_dir(tmp_path: Path) -> None:
    """Un-gzipped capture dirs are parseable (discover_segments takes *.jsonl),
    so detection must claim them too."""
    d = tmp_path / "agent-traces-plain"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", [_dynamo_record()])
    _write_jsonl(d / "b.jsonl", [_dynamo_record(event_type="tool_end")])
    assert detect_format(d) == "dynamo_trace"


def test_plain_jsonl_dir_with_foreign_records_not_claimed(tmp_path: Path) -> None:
    """The dir branch still sniffs the first record: foreign .jsonl dirs fall
    through instead of being claimed by extension alone."""
    d = tmp_path / "not-dynamo"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", [{"schema": "other.v1", "event_type": "request_end"}])
    with pytest.raises(WorkloadFormatError):
        detect_format(d)


# valid 10-byte gzip header + non-deflate payload -> zlib.error on read.
_GZ_HEADER_PLUS_GARBAGE = (
    b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"this-is-not-deflate-data"
)


def test_can_load_corrupt_gzip_returns_false(tmp_path: Path) -> None:
    """zlib.error from corrupt deflate bytes means "not ours", never a crash."""
    from aiperf.dataset.graph.adapters.dynamo.trace import DynamoTraceAdapter

    p = tmp_path / "trace.jsonl.gz"
    p.write_bytes(_GZ_HEADER_PLUS_GARBAGE)
    assert DynamoTraceAdapter.can_load(p) is False


def test_can_load_truncated_gzip_returns_false(tmp_path: Path) -> None:
    """A gz truncated before the first line completes must not crash sniffing."""
    from aiperf.dataset.graph.adapters.dynamo.trace import DynamoTraceAdapter

    p = tmp_path / "trace.jsonl.gz"
    _write_jsonl_gz(p, [_dynamo_record()])
    p.write_bytes(p.read_bytes()[:20])  # header + a few deflate bytes
    assert DynamoTraceAdapter.can_load(p) is False


def test_detect_format_contains_adapter_can_load_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One adapter's can_load raising must not abort detection for the rest."""
    from aiperf.dataset.graph.adapters.dynamo.trace import DynamoTraceAdapter

    def _boom(cls: type, path: Path) -> bool:
        raise RuntimeError("sniff crashed")

    monkeypatch.setattr(DynamoTraceAdapter, "can_load", classmethod(_boom))
    p = tmp_path / "some.jsonl"
    _write_jsonl(p, [{"kind": "note", "text": "not a dynamo record"}])
    assert detect_format(p) == "native"


def test_empty_agent_context_still_routes_dynamo(tmp_path: Path) -> None:
    """The predicate just checks isinstance(dict); empty dict is fine."""
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, [_dynamo_record(agent_context={})])
    assert detect_format(p) == "dynamo_trace"


def test_wrong_schema_falls_through(tmp_path: Path) -> None:
    """A different schema string should not match dynamo_trace."""
    p = tmp_path / "other.jsonl"
    _write_jsonl(
        p,
        [
            {
                "schema": "other.schema.v1",
                "event_type": "request_end",
                "agent_context": {},
            }
        ],
    )
    # Falls through every JSONL predicate and lands on "native".
    assert detect_format(p) == "native"


# A genuine minimal weka trace object: the exact discriminator key set weka's
# file sniff requires ({id, models, block_size, hash_id_scope, requests}).
_WEKA_TRACE_DOC: dict[str, Any] = {
    "id": "trace_ordering_guard",
    "models": ["m"],
    "block_size": 64,
    "hash_id_scope": "local",
    "requests": [{"t": 0.0, "type": "n", "model": "m", "in": 10, "out": 5}],
}


def test_dynamo_predicate_does_not_eat_weka_trace(tmp_path: Path) -> None:
    """Detection ordering: a genuine weka trace routes to weka_trace.

    dynamo_trace has the HIGHER detection_priority (100 vs weka's 85), so if
    its predicate ever claimed a weka trace file it would win the tie-break
    and silently reroute the workload. Lock both halves: dynamo's can_load
    rejects the file, and full registry detection resolves it to weka_trace.
    """
    from aiperf.dataset.graph.adapters.dynamo.trace import DynamoTraceAdapter

    p = tmp_path / "trace_01.json"
    p.write_text(json.dumps(_WEKA_TRACE_DOC))
    assert DynamoTraceAdapter.can_load(p) is False
    assert detect_format(p) == "weka_trace"


def test_dynamo_predicate_does_not_eat_native_graph_jsonl(tmp_path: Path) -> None:
    """Detection ordering: a native graph .jsonl doc falls through to native.

    Native graph workloads share dynamo's ``.jsonl`` extension; dynamo's
    higher-priority predicate must reject the doc so detection lands on the
    lowest-priority native fallback instead of hijacking it.
    """
    from aiperf.dataset.graph.adapters.dynamo.trace import DynamoTraceAdapter

    p = tmp_path / "workload.jsonl"
    _write_jsonl(
        p,
        [
            {
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
        ],
    )
    assert DynamoTraceAdapter.can_load(p) is False
    assert detect_format(p) == "native"


def test_unknown_event_type_falls_through(tmp_path: Path) -> None:
    """Right schema but wrong event_type — does not match dynamo_trace."""
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            {
                "schema": "dynamo.request.trace.v1",
                "event_type": "stream_chunk",  # not in the known set
                "agent_context": {},
            }
        ],
    )
    assert detect_format(p) == "native"


def test_unknown_directory_raises(tmp_path: Path) -> None:
    d = tmp_path / "mystery-dir"
    d.mkdir()
    (d / "unrelated.txt").write_text("hello")
    with pytest.raises(WorkloadFormatError):
        detect_format(d)


def test_parser_dispatch_routes_to_from_dynamo_trace(tmp_path: Path) -> None:
    """Ensure parser.parse_graph dispatches dynamo_trace → from_dynamo_trace.

    Gated on the adapter module's existence — parallel agents are building
    `dynamo/trace.py`. When their commits land on the merge target this test
    will execute against the real adapter; until then it skips at import time.
    """
    pytest.importorskip("aiperf.dataset.graph.adapters.dynamo.trace")

    from aiperf.dataset.graph import parser as parser_mod

    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, [_dynamo_record()])

    called: dict[str, Any] = {}

    def _fake_from_dynamo_trace(path: Path, **kwargs):
        called["path"] = Path(path)
        from aiperf.dataset.graph.models import GraphRecord, ParsedGraph

        return ParsedGraph(
            graph=GraphRecord(),
            traces=[],
        )

    import aiperf.dataset.graph.adapters.dynamo.trace as dt_mod

    monkey_orig = dt_mod.from_dynamo_trace  # type: ignore[attr-defined]
    dt_mod.from_dynamo_trace = _fake_from_dynamo_trace  # type: ignore[attr-defined]
    try:
        parser_mod.parse_graph(p)
    finally:
        dt_mod.from_dynamo_trace = monkey_orig  # type: ignore[attr-defined]

    assert called["path"] == p
