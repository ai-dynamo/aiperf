# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Dynamo agent-trace reader."""

from __future__ import annotations

import gzip
from pathlib import Path

import orjson
import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    AgentContext,
    AgentReplayMetrics,
    AgentTraceRecord,
    DynamoTraceReadError,
    discover_segments,
    iter_raw_records,
    iter_trace_records,
)


def _request_end(
    *,
    event_time_unix_ms: int = 1_000,
    session_id: str = "sess-1",
    request_id: str = "req-1",
    model: str = "my-model",
    replay: dict | None = None,
) -> dict:
    rec: dict = {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": event_time_unix_ms,
        "event_source": "dynamo",
        "agent_context": {
            "session_id": session_id,
        },
        "request": {"request_id": request_id, "model": model},
    }
    if replay is not None:
        rec["request"]["replay"] = replay
    return rec


def _tool_event(
    *,
    event_type: str,
    event_time_unix_ms: int = 1_500,
    session_id: str = "sess-1",
    tool_call_id: str = "call-1",
    tool_class: str = "web_search",
    status: str | None = None,
) -> dict:
    rec: dict = {
        "schema": "dynamo.request.trace.v1",
        "event_type": event_type,
        "event_time_unix_ms": event_time_unix_ms,
        "event_source": "harness",
        "agent_context": {
            "session_id": session_id,
        },
        "tool": {"tool_call_id": tool_call_id, "tool_class": tool_class},
    }
    if status is not None:
        rec["tool"]["status"] = status
    return rec


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def test_iter_plain_jsonl_round_trips_three_records(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _request_end(),
            _tool_event(event_type="tool_start"),
            _tool_event(event_type="tool_end", status="succeeded"),
        ],
    )
    out = list(iter_trace_records(p))
    assert len(out) == 3
    assert out[0].event_type == "request_end"
    assert out[0].request is not None
    assert out[0].request.model == "my-model"
    assert out[1].event_type == "tool_start"
    assert out[1].tool is not None and out[1].tool.tool_call_id == "call-1"
    assert out[2].event_type == "tool_end"
    assert out[2].tool is not None and out[2].tool.status == "succeeded"


def test_iter_gzipped_jsonl_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl.gz"
    _write_jsonl_gz(
        p,
        [
            _request_end(),
            _tool_event(event_type="tool_start"),
            _tool_event(event_type="tool_end", status="succeeded"),
        ],
    )
    out = list(iter_trace_records(p))
    assert [r.event_type for r in out] == ["request_end", "tool_start", "tool_end"]


def test_segmented_directory_iterates_in_segment_order(tmp_path: Path) -> None:
    seg_dir = tmp_path / "segdir"
    seg_dir.mkdir()
    _write_jsonl_gz(seg_dir / "trace.000000.jsonl.gz", [_request_end(request_id="r0")])
    _write_jsonl_gz(seg_dir / "trace.000001.jsonl.gz", [_request_end(request_id="r1")])
    _write_jsonl_gz(seg_dir / "trace.000002.jsonl.gz", [_request_end(request_id="r2")])

    out_dir = list(iter_trace_records(seg_dir))
    assert [r.request.request_id for r in out_dir if r.request] == ["r0", "r1", "r2"]


def test_segmented_prefix_iterates_in_segment_order(tmp_path: Path) -> None:
    seg_dir = tmp_path / "segdir"
    seg_dir.mkdir()
    _write_jsonl_gz(seg_dir / "trace.000000.jsonl.gz", [_request_end(request_id="r0")])
    _write_jsonl_gz(seg_dir / "trace.000001.jsonl.gz", [_request_end(request_id="r1")])
    _write_jsonl_gz(seg_dir / "trace.000002.jsonl.gz", [_request_end(request_id="r2")])

    prefix = seg_dir / "trace"
    segments = discover_segments(prefix)
    assert [s.name for s in segments] == [
        "trace.000000.jsonl.gz",
        "trace.000001.jsonl.gz",
        "trace.000002.jsonl.gz",
    ]
    out_pref = list(iter_trace_records(prefix))
    assert [r.request.request_id for r in out_pref if r.request] == ["r0", "r1", "r2"]


@pytest.mark.parametrize(
    "event_filter,expected_count",
    [
        param({"request_end"}, 2, id="request_end_only"),
        param({"tool_start", "tool_end"}, 2, id="tool_pair"),
        param({"tool_error"}, 1, id="tool_error_only"),
    ],
)  # fmt: skip
def test_filter_by_event_type(
    tmp_path: Path, event_filter: set[str], expected_count: int
) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _request_end(request_id="a"),
            _tool_event(event_type="tool_start"),
            _request_end(request_id="b"),
            _tool_event(event_type="tool_end", status="succeeded"),
            _tool_event(event_type="tool_error", status="error"),
        ],
    )
    out = list(iter_trace_records(p, event_types=event_filter))
    assert len(out) == expected_count
    assert {r.event_type for r in out} <= event_filter


def test_filter_by_session_id_keeps_subset(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _request_end(session_id="sess-A", request_id="a1"),
            _request_end(session_id="sess-B", request_id="b1"),
            _request_end(session_id="sess-A", request_id="a2"),
            _request_end(session_id="sess-B", request_id="b2"),
        ],
    )
    out = list(iter_trace_records(p, session_id="sess-A"))
    assert [r.request.request_id for r in out if r.request] == ["a1", "a2"]


def test_filter_by_subagent_session_id_keeps_subset(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _request_end(session_id="planner", request_id="p1"),
            _request_end(session_id="researcher", request_id="r1"),
            _request_end(session_id="planner", request_id="p2"),
        ],
    )
    out = list(iter_trace_records(p, session_id="planner"))
    assert [r.request.request_id for r in out if r.request] == ["p1", "p2"]


def test_filter_by_time_range_inclusive(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _request_end(event_time_unix_ms=500, request_id="t500"),
            _request_end(event_time_unix_ms=1000, request_id="t1000"),
            _request_end(event_time_unix_ms=2000, request_id="t2000"),
            _request_end(event_time_unix_ms=3000, request_id="t3000"),
            _request_end(event_time_unix_ms=4000, request_id="t4000"),
        ],
    )
    out = list(iter_trace_records(p, time_range_ms=(1000, 3000)))
    assert [r.request.request_id for r in out if r.request] == [
        "t1000",
        "t2000",
        "t3000",
    ]


def test_schema_mismatch_raises(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    bad = _request_end()
    bad["schema"] = "other.schema.v1"
    _write_jsonl(p, [bad])
    with pytest.raises(DynamoTraceReadError):
        list(iter_trace_records(p))


def test_replay_metrics_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    big_hash = 14_879_255_164_371_896_291  # > 2^63, valid u64 from Dynamo
    _write_jsonl(
        p,
        [
            _request_end(
                replay={
                    "trace_block_size": 64,
                    "input_length": 128,
                    "input_sequence_hashes": [1, 2, 3, big_hash],
                }
            )
        ],
    )
    out = list(iter_trace_records(p))
    assert len(out) == 1
    assert out[0].request is not None
    replay = out[0].request.replay
    assert replay is not None
    assert replay.trace_block_size == 64
    assert replay.input_length == 128
    assert replay.input_sequence_hashes == [1, 2, 3, big_hash]


def test_negative_replay_hash_rejected(tmp_path: Path) -> None:
    """A negative recorded hash collides with the virtual negative-id namespace."""
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _request_end(
                replay={
                    "trace_block_size": 16,
                    "input_length": 32,
                    "input_sequence_hashes": [1, -2, 3],
                }
            )
        ],
    )
    with pytest.raises(DynamoTraceReadError):
        list(iter_trace_records(p))

    # Model-level rejection is direct (pydantic ValidationError before the reader
    # wraps it) so callers building AgentReplayMetrics get the same guard.
    with pytest.raises(ValueError, match="non-negative"):
        AgentReplayMetrics(
            trace_block_size=16,
            input_length=32,
            input_sequence_hashes=[1, -2, 3],
        )


def test_schema_alias_round_trip_via_model_dump() -> None:
    rec = _request_end()
    parsed = AgentTraceRecord.model_validate(rec)
    dumped = parsed.model_dump(by_alias=True, exclude_none=True)
    assert dumped["schema"] == "dynamo.request.trace.v1"
    assert "schema_" not in dumped


def test_directory_with_no_trace_files_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DynamoTraceReadError):
        discover_segments(empty)


def test_segmented_prefix_no_match_raises(tmp_path: Path) -> None:
    seg_dir = tmp_path / "segdir"
    seg_dir.mkdir()
    with pytest.raises(DynamoTraceReadError):
        discover_segments(seg_dir / "missing-prefix")


def test_invalid_json_line_raises(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    p.write_bytes(b'{"schema": "dynamo.request.trace.v1"}\nnot-json\n')
    with pytest.raises(DynamoTraceReadError):
        list(iter_raw_records(p))


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    payload = (
        orjson.dumps(_request_end())
        + b"\n\n   \n"
        + orjson.dumps(_request_end(request_id="r2"))
        + b"\n"
    )
    p.write_bytes(payload)
    out = list(iter_trace_records(p))
    assert len(out) == 2


def test_agent_context_required_fields_missing_raises(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    rec = _request_end()
    del rec["agent_context"]["session_id"]
    _write_jsonl(p, [rec])
    with pytest.raises(DynamoTraceReadError):
        list(iter_trace_records(p))


def test_agent_context_empty_string_passes_consumer_side(tmp_path: Path) -> None:
    """Consumer-side parity note: Dynamo's deserialize_non_empty_string is server-side
    only. Empty strings pass here — this test pins that behavior."""
    p = tmp_path / "trace.jsonl"
    rec = _request_end()
    rec["agent_context"]["session_id"] = ""
    _write_jsonl(p, [rec])
    out = list(iter_trace_records(p))
    assert len(out) == 1
    assert out[0].agent_context.session_id == ""


def test_extra_fields_ignored() -> None:
    rec = _request_end()
    rec["future_field"] = {"new": "thing"}
    rec["request"]["another_future"] = 42
    parsed = AgentTraceRecord.model_validate(rec)
    assert not hasattr(parsed, "future_field")
    assert parsed.request is not None


def test_agent_context_parent_session_id_optional() -> None:
    ac = AgentContext(session_id="z")
    assert ac.parent_session_id is None
    ac2 = AgentContext(
        session_id="z",
        parent_session_id="parent",
    )
    assert ac2.parent_session_id == "parent"


# --- dynamo file-sink envelope + discovery parity -----------------------------


def _wrap(rec: dict, ts: int = 12) -> dict:
    """Wrap a record the way both dynamo file sinks do (telemetry/jsonl_gz.rs)."""
    return {"timestamp": ts, "event": rec}


def test_sink_envelope_unwrapped_plain_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, [_wrap(_request_end(request_id="req-w1"))])
    out = list(iter_trace_records(p))
    assert len(out) == 1
    assert out[0].request is not None and out[0].request.request_id == "req-w1"


def test_sink_envelope_unwrapped_gz_segments(tmp_path: Path) -> None:
    _write_jsonl_gz(
        tmp_path / "trace.000000.jsonl.gz",
        [_wrap(_request_end(event_time_unix_ms=1_000, request_id="a"))],
    )
    _write_jsonl_gz(
        tmp_path / "trace.000001.jsonl.gz",
        [_wrap(_request_end(event_time_unix_ms=2_000, request_id="b"), ts=99)],
    )
    out = list(iter_trace_records(tmp_path / "trace"))
    assert [r.request.request_id for r in out] == ["a", "b"]


def test_bare_and_wrapped_records_mix(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p, [_request_end(request_id="bare"), _wrap(_request_end(request_id="wrapped"))]
    )
    assert [r.request.request_id for r in iter_trace_records(p)] == ["bare", "wrapped"]


def test_prefix_with_output_path_suffix_is_stripped(tmp_path: Path) -> None:
    # DYN_REQUEST_TRACE_OUTPUT_PATH=/x/trace.jsonl.gz produces /x/trace.000000.jsonl.gz;
    # passing the configured path verbatim must resolve the segments.
    _write_jsonl_gz(tmp_path / "trace.000000.jsonl.gz", [_request_end()])
    for given in ("trace.jsonl.gz", "trace.jsonl", "trace"):
        segs = discover_segments(tmp_path / given)
        assert [s.name for s in segs] == ["trace.000000.jsonl.gz"]


def test_directory_segment_order_is_numeric_past_six_digits(tmp_path: Path) -> None:
    _write_jsonl_gz(
        tmp_path / "t.1000000.jsonl.gz", [_request_end(event_time_unix_ms=3_000)]
    )
    _write_jsonl_gz(
        tmp_path / "t.999999.jsonl.gz", [_request_end(event_time_unix_ms=2_000)]
    )
    _write_jsonl_gz(
        tmp_path / "t.000001.jsonl.gz", [_request_end(event_time_unix_ms=1_000)]
    )
    segs = discover_segments(tmp_path)
    assert [s.name for s in segs] == [
        "t.000001.jsonl.gz",
        "t.999999.jsonl.gz",
        "t.1000000.jsonl.gz",
    ]


def test_truncated_final_gzip_member_raises_read_error(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl.gz"
    _write_jsonl_gz(p, [_request_end()])
    data = p.read_bytes()
    p.write_bytes(data[: len(data) - 8])  # chop the gzip trailer mid-member
    with pytest.raises(DynamoTraceReadError, match="truncated or corrupt gzip"):
        list(iter_raw_records(p))


def test_trace_block_size_zero_rejected() -> None:
    with pytest.raises(ValueError, match="trace_block_size"):
        AgentReplayMetrics(
            trace_block_size=0, input_length=4, input_sequence_hashes=[1]
        )


# valid 10-byte gzip header followed by bytes that are not a deflate stream:
# reading raises zlib.error (not EOFError / BadGzipFile).
_GZ_HEADER_PLUS_GARBAGE = (
    b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"this-is-not-deflate-data"
)


def test_corrupt_gzip_deflate_raises_read_error(tmp_path: Path) -> None:
    """zlib.error from corrupt deflate data must wrap into DynamoTraceReadError."""
    p = tmp_path / "trace.jsonl.gz"
    p.write_bytes(_GZ_HEADER_PLUS_GARBAGE)
    with pytest.raises(DynamoTraceReadError, match="truncated or corrupt gzip"):
        list(iter_raw_records(p))


def test_non_utf8_jsonl_raises_read_error(tmp_path: Path) -> None:
    """Binary bytes behind a .jsonl name must wrap into DynamoTraceReadError."""
    p = tmp_path / "trace.jsonl"
    p.write_bytes(b"\xff\xfe\x00garbage-bytes")
    with pytest.raises(DynamoTraceReadError, match="not valid UTF-8"):
        list(iter_raw_records(p))


def test_uppercase_gz_suffix_opens_as_gzip(tmp_path: Path) -> None:
    """Detection lowercases suffixes; the segment opener must match."""
    p = tmp_path / "trace.JSONL.GZ"
    _write_jsonl_gz(p, [_request_end(request_id="upper")])
    out = list(iter_trace_records(p))
    assert [r.request.request_id for r in out if r.request] == ["upper"]
